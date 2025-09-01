"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import os
import asyncio
import base64
import io
import traceback
import re
import time
import json
from datetime import datetime

import cv2
import pyaudio
import PIL.Image
import mss
import numpy as np

import argparse

from google import genai
from google.genai import types
from dtmf_generator import DTMFGenerator

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "camera"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Airline information database
AIRLINE_INFO = {
    "american": {"name": "American Airlines", "phone": "1-800-433-7300", "code": "AA"},
    "delta": {"name": "Delta Airlines", "phone": "1-800-221-1212", "code": "DL"},
    "united": {"name": "United Airlines", "phone": "1-800-864-8331", "code": "UA"},
    "southwest": {"name": "Southwest Airlines", "phone": "1-800-435-9792", "code": "WN"},
    "jetblue": {"name": "JetBlue Airways", "phone": "1-800-538-2583", "code": "B6"},
}

# Common IVR menu patterns
COMMON_MENU_PATTERNS = {
    "flight_status": ["1", "2", "flight", "status"],
    "existing_reservation": ["2", "3", "existing", "reservation", "booking"],
    "customer_service": ["0", "9", "agent", "representative", "customer", "service"],
    "baggage": ["3", "4", "baggage", "luggage"],
    "check_in": ["1", "check", "in"],
    "changes_cancellation": ["4", "5", "change", "cancel", "modification"]
}

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="press_number",
                description="Press a number (0-9) or symbol (*,#) on the phone keypad to navigate IVR menus. This generates DTMF tones that the IVR system will respond to.",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "digit": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The digit or symbol to press (0-9, *, #)"
                        ),
                        "reason": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="Why you're pressing this digit (for logging)"
                        ),
                    },
                    required=["digit"]
                ),
            ),
            types.FunctionDeclaration(
                name="provide_flight_info",
                description="Provide flight information when prompted by the IVR system (confirmation number, flight number, etc.)",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "info_type": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="Type of information being provided",
                            enum=["flight_number", "confirmation_number", "passenger_name", "phone_number", "date"]
                        ),
                        "value": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The actual information value"
                        ),
                    },
                    required=["info_type", "value"]
                ),
            ),
            types.FunctionDeclaration(
                name="wait_for_menu",
                description="Wait for the IVR menu options to be fully announced before taking action",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "duration": genai.types.Schema(
                            type=genai.types.Type.INTEGER,
                            description="Seconds to wait (default 3)"
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="log_call_progress",
                description="Log the current status and progress of the call for debugging and user updates",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "status": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="Current status of the call"
                        ),
                        "action": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="Action being taken or completed"
                        ),
                    },
                    required=["status", "action"]
                ),
            ),
            types.FunctionDeclaration(
                name="extract_flight_details",
                description="Extract and save flight information received from the airline",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "flight_info": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The flight information received from the airline"
                        ),
                    },
                    required=["flight_info"]
                ),
            ),
        ]
    )
]

def parse_user_task(task_description):
    """Parse user task and extract relevant information"""
    task_info = {
        "task_type": "unknown",
        "airline": None,
        "flight_number": None,
        "confirmation_number": None,
        "passenger_name": None,
        "date": None
    }
    
    task_lower = task_description.lower()
    
    # Determine task type
    if any(word in task_lower for word in ["flight", "status", "check flight"]):
        task_info["task_type"] = "flight_status"
    elif any(word in task_lower for word in ["reservation", "booking", "confirm"]):
        task_info["task_type"] = "reservation_info"
    elif any(word in task_lower for word in ["change", "modify", "cancel"]):
        task_info["task_type"] = "change_reservation"
    elif any(word in task_lower for word in ["baggage", "luggage"]):
        task_info["task_type"] = "baggage_inquiry"
    
    # Extract airline
    for key, info in AIRLINE_INFO.items():
        if key in task_lower or info["code"].lower() in task_lower:
            task_info["airline"] = key
            break
    
    # Extract flight number
    flight_pattern = r'([A-Z]{1,3})\s*(\d{1,4})'
    flight_match = re.search(flight_pattern, task_description.upper())
    if flight_match:
        task_info["flight_number"] = f"{flight_match.group(1)}{flight_match.group(2)}"
    
    # Extract confirmation number (typically 6 characters alphanumeric)
    conf_pattern = r'\b([A-Z0-9]{6})\b'
    conf_match = re.search(conf_pattern, task_description.upper())
    if conf_match:
        task_info["confirmation_number"] = conf_match.group(1)
    
    return task_info

def create_agent_instructions(task_info, user_instructions):
    """Create detailed instructions for the AI agent based on the task"""
    
    base_instructions = """You are an autonomous phone agent designed to call airline customer service and complete specific tasks. 

IMPORTANT BEHAVIORAL GUIDELINES:
1. Be patient and wait for menu options to be fully announced
2. Press numbers/keys deliberately and wait for confirmation
3. Speak clearly and at normal pace when providing information
4. Use the function tools to navigate menus and provide information
5. Log your progress regularly so the user can track what's happening
6. If you get stuck or hear unexpected options, try pressing '0' for customer service
7. Listen carefully to all menu options before making choices
8. Stay focused on the specific task - don't get sidetracked

YOUR CURRENT TASK:"""
    
    task_type = task_info.get("task_type", "unknown")
    airline = task_info.get("airline")
    flight_number = task_info.get("flight_number")
    confirmation_number = task_info.get("confirmation_number")
    
    specific_instructions = ""
    
    if task_type == "flight_status":
        specific_instructions = f"""
TASK: Check flight status
Flight Number: {flight_number or 'Not provided'}
Airline: {airline or 'Not provided'}

EXPECTED WORKFLOW:
1. Wait for main menu and listen to all options
2. Select the option for "Flight Status" or "Flight Information" (usually option 1 or 2)
3. When prompted, provide the flight number: {flight_number}
4. Listen to the flight status information provided
5. Use extract_flight_details function to save the information
6. Complete the call

MENU NAVIGATION HINTS:
- Flight status is typically option 1 or 2
- You may need to select "Today's flights" or choose a date
- Some airlines ask for departure city first
"""
    
    elif task_type == "reservation_info":
        specific_instructions = f"""
TASK: Get reservation information
Confirmation Number: {confirmation_number or 'Not provided'}
Flight Number: {flight_number or 'Not provided'}

EXPECTED WORKFLOW:
1. Wait for main menu and listen to all options
2. Select "Existing Reservation" or "Manage Booking" (usually option 2 or 3)
3. Provide confirmation number when prompted: {confirmation_number}
4. Listen to the reservation details
5. Use extract_flight_details function to save the information
"""
    
    additional_context = f"""
AIRLINE INFORMATION:
{f"Calling: {AIRLINE_INFO[airline]['name']} at {AIRLINE_INFO[airline]['phone']}" if airline and airline in AIRLINE_INFO else "Airline not specified"}

USER'S ADDITIONAL INSTRUCTIONS:
{user_instructions}

FUNCTION USAGE:
- Use press_number() to navigate menus
- Use provide_flight_info() when asked for flight numbers, confirmation codes, etc.
- Use wait_for_menu() if you need to pause and listen
- Use log_call_progress() to keep the user informed
- Use extract_flight_details() to save any flight information you receive
"""
    
    return base_instructions + specific_instructions + additional_context

def create_config(user_instructions, task_info=None):
    """Create enhanced config with task-specific instructions"""
    if task_info:
        full_instructions = create_agent_instructions(task_info, user_instructions)
    else:
        full_instructions = f"You are a helpful assistant designed to take calls on behalf of the user and follow the instructions provided by the user. \n\n\n USER INSTRUCTIONS:{user_instructions}"
        
    return types.LiveConnectConfig(
        response_modalities=[
            "AUDIO",
        ],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=types.SpeechConfig(
            language_code="en-US",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
        tools=tools,
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=full_instructions)],
            role="user"
        ),
    )

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, user_instructions="", task=None):
        self.video_mode = video_mode
        self.user_instructions = user_instructions
        self.task = task
        self.task_info = None
        
        # Initialize DTMF generator
        self.dtmf_generator = DTMFGenerator()
        
        # Parse task if provided
        if task:
            self.task_info = parse_user_task(task)
            print(f"\n🤖 PARSED TASK INFO:")
            print(f"   Task Type: {self.task_info['task_type']}")
            print(f"   Airline: {self.task_info['airline']}")
            print(f"   Flight Number: {self.task_info['flight_number']}")
            print(f"   Confirmation: {self.task_info['confirmation_number']}")
            if self.task_info['airline'] and self.task_info['airline'] in AIRLINE_INFO:
                airline_info = AIRLINE_INFO[self.task_info['airline']]
                print(f"   📞 Will call: {airline_info['name']} at {airline_info['phone']}")
            print()

        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.call_log = []
        
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    def __del__(self):
        """Cleanup DTMF generator when object is destroyed"""
        if hasattr(self, 'dtmf_generator'):
            self.dtmf_generator.close()

    def log_call_event(self, event_type, message):
        """Log call events for debugging and user feedback"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {event_type}: {message}"
        self.call_log.append(log_entry)
        print(f"📱 {log_entry}")

    async def handle_function_call(self, function_name, parameters):
        """Handle function calls from the AI agent"""
        try:
            if function_name == "press_number":
                digit = parameters.get("digit")
                reason = parameters.get("reason", "Menu navigation")
                self.log_call_event("DTMF", f"Pressing '{digit}' - {reason}")
                
                # Generate and play actual DTMF tone
                await self.dtmf_generator.play_tone(digit)
                await asyncio.sleep(0.3)  # Brief pause after tone
                
            elif function_name == "provide_flight_info":
                info_type = parameters.get("info_type")
                value = parameters.get("value")
                self.log_call_event("INFO", f"Providing {info_type}: {value}")
                
                # For numeric info (flight numbers, confirmation codes), 
                # we might need to press DTMF tones
                if info_type in ["flight_number", "confirmation_number"] and value:
                    # Extract numeric/alphanumeric parts that can be entered via keypad
                    numeric_part = re.sub(r'[^0-9*#]', '', value.upper())
                    if numeric_part:
                        self.log_call_event("DTMF", f"Entering {info_type} digits: {numeric_part}")
                        await self.dtmf_generator.play_sequence(numeric_part, pause_between=0.2)
                
            elif function_name == "wait_for_menu":
                duration = parameters.get("duration", 3)
                self.log_call_event("WAIT", f"Waiting {duration} seconds for menu")
                await asyncio.sleep(duration)
                
            elif function_name == "log_call_progress":
                status = parameters.get("status")
                action = parameters.get("action")
                self.log_call_event("STATUS", f"{status} - {action}")
                
            elif function_name == "extract_flight_details":
                flight_info = parameters.get("flight_info")
                self.log_call_event("RESULT", f"Flight info received: {flight_info}")
                
                # Save to file for user reference
                result_file = f"flight_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(result_file, 'w') as f:
                    f.write(f"Flight Information Retrieved:\n")
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Task: {self.task}\n")
                    f.write(f"Details: {flight_info}\n")
                    f.write(f"\nCall Log:\n")
                    for entry in self.call_log:
                        f.write(f"{entry}\n")
                
                print(f"✅ Flight information saved to {result_file}")
                
        except Exception as e:
            self.log_call_event("ERROR", f"Function call failed: {str(e)}")

    async def send_text(self):
        """Enhanced text input that can also trigger autonomous mode"""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            elif text.lower().startswith("auto:"):
                # Trigger autonomous mode with a new task
                task_description = text[5:].strip()
                task_info = parse_user_task(task_description)
                instructions = create_agent_instructions(task_info, self.user_instructions)
                await self.session.send(input=f"NEW TASK: {instructions}", end_of_turn=True)
            elif text.lower() == "status":
                # Show current call log
                print("\n📋 CALL LOG:")
                for entry in self.call_log[-10:]:  # Show last 10 entries
                    print(f"   {entry}")
                continue
            else:
                await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                # Handle function calls
                if hasattr(response, 'function_call') and response.function_call:
                    func_call = response.function_call
                    await self.handle_function_call(func_call.name, func_call.parameters)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            # Create config with task-specific instructions
            config = create_config(self.user_instructions, self.task_info)
            
            print("🚀 Starting AI Phone Agent...")
            if self.task:
                print(f"📋 Task: {self.task}")
                if self.task_info and self.task_info['airline'] in AIRLINE_INFO:
                    airline_info = AIRLINE_INFO[self.task_info['airline']]
                    print(f"📞 Ready to call: {airline_info['name']}")
                print("💡 The agent will autonomously navigate the phone system")
                print("💡 Type 'status' to see call progress, 'q' to quit")
                print("💡 Type 'auto: <task>' to give the agent a new task")
            
            async with (
                client.aio.live.connect(model=MODEL, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # If we have a task, start autonomous mode
                if self.task:
                    # Give the agent the initial context about the call
                    initial_message = f"I am now ready to start the call. My task is: {self.task}. Please begin by acknowledging the task and then start navigating the phone system according to your instructions."
                    await self.session.send(input=initial_message, end_of_turn=True)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            print("\n📞 Call ended by user")
            if self.call_log:
                print("\n📋 Final Call Summary:")
                for entry in self.call_log:
                    print(f"   {entry}")
        except ExceptionGroup as EG:
            if hasattr(self, 'audio_stream'):
                self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Phone Agent for 1800 numbers")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Video mode for the agent",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="",
        help="Additional instructions for the AI assistant",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Specific task for the agent (e.g., 'check flight status for AA1123', 'get reservation info for ABC123')"
    )
    
    # Add some example usage
    parser.epilog = """
    EXAMPLE USAGE:
    
    Flight Status Check:
        python app.py --task "check flight status AA1123"
        python app.py --task "flight status Delta 2456"
    
    Reservation Info:
        python app.py --task "get reservation info ABC123"
        python app.py --task "check booking confirmation XYZ789"
    
    Custom Task:
        python app.py --task "call United about baggage claim" --instructions "Be polite and ask for supervisor if needed"
    
    Interactive Mode:
        python app.py --mode none
        Then type: auto: check my American flight AA1123
    """
    
    args = parser.parse_args()
    
    if args.task:
        print("=" * 60)
        print("🤖 AI PHONE AGENT - AUTONOMOUS MODE")
        print("=" * 60)
    else:
        print("=" * 60) 
        print("🤖 AI PHONE AGENT - INTERACTIVE MODE")
        print("=" * 60)
        print("💡 Type 'auto: <task>' to give the agent an autonomous task")
        print("💡 Example: auto: check flight status AA1123")
    
    main = AudioLoop(
        video_mode=args.mode, 
        user_instructions=args.instructions,
        task=args.task
    )
    asyncio.run(main.run())
