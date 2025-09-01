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
import logging
from datetime import datetime
from typing import Optional

import cv2
import pyaudio
import PIL.Image
import mss
import numpy as np

import argparse

from google import genai
from google.genai import types
from dtmf_generator import DTMFGenerator

# Configure logging
def setup_logging(log_level="INFO", log_file=None):
    """Setup comprehensive logging system"""
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Global logger
logger = logging.getLogger(__name__)

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
    def __init__(self, video_mode=DEFAULT_MODE, user_instructions="", task=None, log_level="INFO"):
        self.logger = logging.getLogger(f"{__name__}.AudioLoop")
        self.logger.info("Initializing AI Phone Agent...")
        
        self.video_mode = video_mode
        self.user_instructions = user_instructions
        self.task = task
        self.task_info = None
        self.log_level = log_level
        
        # Initialize DTMF generator
        self.logger.debug("Initializing DTMF generator...")
        self.dtmf_generator = DTMFGenerator()
        
        # Parse task if provided
        if task:
            self.logger.info(f"Parsing user task: {task}")
            self.task_info = parse_user_task(task)
            self.logger.info("Task parsing results:")
            self.logger.info(f"  Task Type: {self.task_info['task_type']}")
            self.logger.info(f"  Airline: {self.task_info['airline']}")
            self.logger.info(f"  Flight Number: {self.task_info['flight_number']}")
            self.logger.info(f"  Confirmation: {self.task_info['confirmation_number']}")
            
            if self.task_info['airline'] and self.task_info['airline'] in AIRLINE_INFO:
                airline_info = AIRLINE_INFO[self.task_info['airline']]
                self.logger.info(f"  Target: {airline_info['name']} at {airline_info['phone']}")
                
                print(f"\n🤖 PARSED TASK INFO:")
                print(f"   Task Type: {self.task_info['task_type']}")
                print(f"   Airline: {self.task_info['airline']}")
                print(f"   Flight Number: {self.task_info['flight_number']}")
                print(f"   Confirmation: {self.task_info['confirmation_number']}")
                print(f"   📞 Will call: {airline_info['name']} at {airline_info['phone']}")
                print()

        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.call_log = []
        self.session_start_time = None
        
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        
        self.logger.debug("AudioLoop initialization complete")

    def __del__(self):
        """Cleanup DTMF generator when object is destroyed"""
        if hasattr(self, 'dtmf_generator'):
            self.logger.debug("Cleaning up DTMF generator")
            self.dtmf_generator.close()

    def log_call_event(self, event_type, message, level="INFO"):
        """Enhanced logging for call events"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        session_duration = ""
        
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            session_duration = f" [{duration.total_seconds():.1f}s]"
        
        log_entry = f"[{timestamp}{session_duration}] {event_type}: {message}"
        self.call_log.append(log_entry)
        
        # Log to both console and file with appropriate level
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"{event_type}: {message}")
        
        # Console output with emojis for key events
        emoji_map = {
            "DTMF": "📞",
            "INFO": "ℹ️",
            "WAIT": "⏳", 
            "STATUS": "📊",
            "RESULT": "✅",
            "ERROR": "❌",
            "SESSION": "🔗",
            "AUDIO": "🔊",
            "SYSTEM": "⚙️"
        }
        
        emoji = emoji_map.get(event_type, "📱")
        print(f"{emoji} {log_entry}")

    def get_call_statistics(self):
        """Get call statistics and summary"""
        stats = {
            "total_events": len(self.call_log),
            "dtmf_presses": len([e for e in self.call_log if "DTMF:" in e]),
            "errors": len([e for e in self.call_log if "ERROR:" in e]),
            "duration": None
        }
        
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            stats["duration"] = duration.total_seconds()
            
        return stats

    async def handle_function_call(self, function_name, parameters):
        """Handle function calls from the AI agent with detailed logging"""
        self.logger.debug(f"Handling function call: {function_name} with params: {parameters}")
        
        try:
            if function_name == "press_number":
                digit = parameters.get("digit")
                reason = parameters.get("reason", "Menu navigation")
                self.logger.info(f"AI pressing DTMF key: '{digit}' - {reason}")
                self.log_call_event("DTMF", f"Pressing '{digit}' - {reason}")
                
                # Generate and play actual DTMF tone
                self.logger.debug(f"Generating DTMF tone for digit: {digit}")
                await self.dtmf_generator.play_tone(digit)
                self.logger.debug("DTMF tone playback complete")
                await asyncio.sleep(0.3)  # Brief pause after tone
                
            elif function_name == "provide_flight_info":
                info_type = parameters.get("info_type")
                value = parameters.get("value")
                self.logger.info(f"AI providing {info_type}: {value}")
                self.log_call_event("INFO", f"Providing {info_type}: {value}")
                
                # For numeric info (flight numbers, confirmation codes), 
                # we might need to press DTMF tones
                if info_type in ["flight_number", "confirmation_number"] and value:
                    # Extract numeric/alphanumeric parts that can be entered via keypad
                    numeric_part = re.sub(r'[^0-9*#]', '', value.upper())
                    if numeric_part:
                        self.logger.debug(f"Extracting numeric part: {numeric_part} from {value}")
                        self.log_call_event("DTMF", f"Entering {info_type} digits: {numeric_part}")
                        await self.dtmf_generator.play_sequence(numeric_part, pause_between=0.2)
                        self.logger.debug("DTMF sequence playback complete")
                
            elif function_name == "wait_for_menu":
                duration = parameters.get("duration", 3)
                self.logger.info(f"AI waiting {duration} seconds for menu")
                self.log_call_event("WAIT", f"Waiting {duration} seconds for menu")
                await asyncio.sleep(duration)
                self.logger.debug(f"Wait complete after {duration} seconds")
                
            elif function_name == "log_call_progress":
                status = parameters.get("status")
                action = parameters.get("action")
                self.logger.info(f"Call progress - {status}: {action}")
                self.log_call_event("STATUS", f"{status} - {action}")
                
            elif function_name == "extract_flight_details":
                flight_info = parameters.get("flight_info")
                self.logger.info(f"Extracting flight details: {flight_info}")
                self.log_call_event("RESULT", f"Flight info received: {flight_info}")
                
                # Save to file for user reference
                result_file = f"flight_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                self.logger.debug(f"Saving results to file: {result_file}")
                
                with open(result_file, 'w') as f:
                    f.write(f"Flight Information Retrieved:\n")
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Task: {self.task}\n")
                    f.write(f"Details: {flight_info}\n")
                    f.write(f"\nCall Statistics:\n")
                    stats = self.get_call_statistics()
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"\nDetailed Call Log:\n")
                    for entry in self.call_log:
                        f.write(f"{entry}\n")
                
                self.logger.info(f"Flight information saved to {result_file}")
                print(f"✅ Flight information saved to {result_file}")
                
        except Exception as e:
            self.logger.error(f"Function call failed: {function_name} - {str(e)}")
            self.logger.debug(f"Function call exception details:", exc_info=True)
            self.log_call_event("ERROR", f"Function call failed: {str(e)}", "ERROR")

    async def send_text(self):
        """Enhanced text input with logging and debug commands"""
        self.logger.debug("Starting text input loop")
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            
            self.logger.debug(f"User input received: {text}")
            
            if text.lower() == "q":
                self.logger.info("User requested exit")
                break
            elif text.lower().startswith("auto:"):
                # Trigger autonomous mode with a new task
                task_description = text[5:].strip()
                self.logger.info(f"User requesting autonomous task: {task_description}")
                task_info = parse_user_task(task_description)
                instructions = create_agent_instructions(task_info, self.user_instructions)
                self.log_call_event("SYSTEM", f"New autonomous task: {task_description}")
                await self.session.send(input=f"NEW TASK: {instructions}", end_of_turn=True)
            elif text.lower() == "status":
                # Show current call log and statistics
                print("\n📋 CALL STATUS:")
                stats = self.get_call_statistics()
                print(f"   Duration: {stats['duration']:.1f}s" if stats['duration'] else "   Duration: N/A")
                print(f"   Total Events: {stats['total_events']}")
                print(f"   DTMF Presses: {stats['dtmf_presses']}")
                print(f"   Errors: {stats['errors']}")
                print("\n📜 RECENT LOG (last 10 entries):")
                for entry in self.call_log[-10:]:
                    print(f"   {entry}")
                self.logger.info("User requested status display")
                continue
            elif text.lower() == "debug":
                # Toggle debug logging
                current_level = self.logger.getEffectiveLevel()
                new_level = logging.DEBUG if current_level > logging.DEBUG else logging.INFO
                logging.getLogger().setLevel(new_level)
                level_name = logging.getLevelName(new_level)
                print(f"🔧 Debug logging {'enabled' if new_level == logging.DEBUG else 'disabled'} ({level_name})")
                self.logger.info(f"Logging level changed to {level_name}")
                continue
            elif text.lower() == "logs":
                # Show full call log
                print("\n📜 FULL CALL LOG:")
                for entry in self.call_log:
                    print(f"   {entry}")
                continue
            elif text.lower() == "help":
                # Show help
                print("\n🆘 AVAILABLE COMMANDS:")
                print("   q - Quit the application")
                print("   status - Show call status and recent logs")
                print("   logs - Show full call log")
                print("   debug - Toggle debug logging")
                print("   auto: <task> - Give agent a new autonomous task")
                print("   help - Show this help message")
                continue
            else:
                self.logger.debug(f"Sending user message to AI: {text}")
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
        self.logger.debug("Starting audio input stream")
        mic_info = pya.get_default_input_device_info()
        self.logger.info(f"Using microphone: {mic_info['name']}")
        
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        self.log_call_event("AUDIO", f"Audio input started - {mic_info['name']}")
        
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
            
        audio_chunks_sent = 0
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                audio_chunks_sent += 1
                
                # Log every 100 chunks to avoid spam
                if audio_chunks_sent % 100 == 0:
                    self.logger.debug(f"Sent {audio_chunks_sent} audio chunks")
                    
            except Exception as e:
                self.logger.error(f"Audio input error: {str(e)}")
                self.log_call_event("ERROR", f"Audio input error: {str(e)}", "ERROR")
                break

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        self.logger.debug("Starting audio receive stream")
        audio_chunks_received = 0
        
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        audio_chunks_received += 1
                        
                        # Log every 50 chunks to avoid spam  
                        if audio_chunks_received % 50 == 0:
                            self.logger.debug(f"Received {audio_chunks_received} audio chunks from AI")
                        continue
                        
                    if text := response.text:
                        self.logger.debug(f"AI response text: {text[:100]}...")
                        print(text, end="")
                        
                    # Handle function calls
                    if hasattr(response, 'function_call') and response.function_call:
                        func_call = response.function_call
                        self.logger.info(f"AI requesting function call: {func_call.name}")
                        await self.handle_function_call(func_call.name, func_call.parameters)

                # If you interrupt the model, it sends a turn_complete.
                # For interruptions to work, we need to stop playback.
                # So empty out the audio queue because it may have loaded
                # much more audio than has played yet.
                queue_size = self.audio_in_queue.qsize()
                if queue_size > 0:
                    self.logger.debug(f"Clearing {queue_size} audio chunks from queue")
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                        
            except Exception as e:
                self.logger.error(f"Audio receive error: {str(e)}")
                self.log_call_event("ERROR", f"Audio receive error: {str(e)}", "ERROR")
                break

    async def play_audio(self):
        self.logger.debug("Starting audio output stream")
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        self.log_call_event("AUDIO", "Audio output started")
        audio_chunks_played = 0
        
        while True:
            try:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
                audio_chunks_played += 1
                
                # Log every 50 chunks to avoid spam
                if audio_chunks_played % 50 == 0:
                    self.logger.debug(f"Played {audio_chunks_played} audio chunks")
                    
            except Exception as e:
                self.logger.error(f"Audio playback error: {str(e)}")
                self.log_call_event("ERROR", f"Audio playback error: {str(e)}", "ERROR")
                break

    async def run(self):
        self.session_start_time = datetime.now()
        self.logger.info("Starting AI Phone Agent session")
        self.log_call_event("SESSION", "AI Phone Agent session started")
        
        try:
            # Create config with task-specific instructions
            self.logger.debug("Creating configuration...")
            config = create_config(self.user_instructions, self.task_info)
            
            print("🚀 Starting AI Phone Agent...")
            if self.task:
                print(f"📋 Task: {self.task}")
                if self.task_info and self.task_info['airline'] in AIRLINE_INFO:
                    airline_info = AIRLINE_INFO[self.task_info['airline']]
                    print(f"📞 Ready to call: {airline_info['name']}")
                print("💡 The agent will autonomously navigate the phone system")
                
            print("\n� AVAILABLE COMMANDS:")
            print("   status - Show call progress and statistics")
            print("   logs - Show full call log")
            print("   debug - Toggle debug logging")
            print("   help - Show all commands")
            print("   auto: <task> - Give agent a new task")
            print("   q - Quit")
            
            self.logger.info("Connecting to Gemini Live API...")
            async with (
                client.aio.live.connect(model=MODEL, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.log_call_event("SESSION", "Connected to Gemini Live API")

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # If we have a task, start autonomous mode
                if self.task:
                    self.logger.info("Starting autonomous mode with task")
                    # Give the agent the initial context about the call
                    initial_message = f"I am now ready to start the call. My task is: {self.task}. Please begin by acknowledging the task and then start navigating the phone system according to your instructions."
                    await self.session.send(input=initial_message, end_of_turn=True)
                    self.log_call_event("SYSTEM", f"Autonomous task initiated: {self.task}")

                self.logger.debug("Starting async tasks...")
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    self.logger.info("Starting camera feed")
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    self.logger.info("Starting screen capture")
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            self.logger.info("Session ended by user request")
            print("\n📞 Call ended by user")
            if self.call_log:
                print("\n📋 Final Call Summary:")
                stats = self.get_call_statistics()
                print(f"   Duration: {stats['duration']:.1f}s" if stats['duration'] else "   Duration: N/A")
                print(f"   Total Events: {stats['total_events']}")
                print(f"   DTMF Presses: {stats['dtmf_presses']}")
                print(f"   Errors: {stats['errors']}")
                
                print("\n📜 Call Log:")
                for entry in self.call_log:
                    print(f"   {entry}")
                    
        except ExceptionGroup as EG:
            self.logger.error("Session ended with exception group", exc_info=True)
            if hasattr(self, 'audio_stream'):
                self.audio_stream.close()
            traceback.print_exception(EG)
        except Exception as e:
            self.logger.error(f"Unexpected error in session: {str(e)}", exc_info=True)
            raise


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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional log file to save detailed logs"
    )
    
    # Add some example usage
    parser.epilog = """
    EXAMPLE USAGE:
    
    Flight Status Check:
        python app.py --task "check flight status AA1123"
        python app.py --task "flight status Delta 2456" --log-level DEBUG
    
    Reservation Info:
        python app.py --task "get reservation info ABC123" --log-file call_log.txt
        python app.py --task "check booking confirmation XYZ789"
    
    Custom Task:
        python app.py --task "call United about baggage claim" --instructions "Be polite and ask for supervisor if needed"
    
    Interactive Mode:
        python app.py --mode none --log-level DEBUG
        Then type: auto: check my American flight AA1123
    
    Debug Mode:
        python app.py --task "check AA1123" --log-level DEBUG --log-file debug.log
    """
    
    args = parser.parse_args()
    
    # Setup logging based on arguments
    log_file = args.log_file
    if args.task and not log_file:
        # Auto-generate log file for autonomous tasks
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"ai_agent_log_{timestamp}.log"
        
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    if args.task:
        print("=" * 60)
        print("🤖 AI PHONE AGENT - AUTONOMOUS MODE")
        print("=" * 60)
        logger.info(f"Starting autonomous mode with task: {args.task}")
    else:
        print("=" * 60) 
        print("🤖 AI PHONE AGENT - INTERACTIVE MODE")
        print("=" * 60)
        print("💡 Type 'auto: <task>' to give the agent an autonomous task")
        print("💡 Example: auto: check flight status AA1123")
        logger.info("Starting interactive mode")
    
    if log_file:
        print(f"📝 Detailed logs will be saved to: {log_file}")
        logger.info(f"Logging to file: {log_file}")
    
    main = AudioLoop(
        video_mode=args.mode, 
        user_instructions=args.instructions,
        task=args.task,
        log_level=args.log_level
    )
    
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"\n❌ Application error: {str(e)}")
