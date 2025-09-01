# AI Phone Agent for 1800 Numbers

An autonomous AI agent that can call 1800 numbers, navigate IVR menus, and complete tasks on your behalf using Google's Gemini Live API.

## 🚀 Features

- **Autonomous Navigation**: AI agent can understand and navigate complex phone menus
- **DTMF Tone Generation**: Real DTMF tones for keypad navigation
- **Flight Status Checking**: Built-in knowledge of airline systems
- **Reservation Management**: Can check bookings and confirmations
- **Real-time Logging**: Track the agent's progress through the call
- **Multi-modal Input**: Audio, video, and text interactions

## 📋 Prerequisites

1. **Google Gemini API Key**: Set your `GEMINI_API_KEY` environment variable
2. **Python Dependencies**: Install required packages
3. **Audio Setup**: Ensure microphone and speakers are working

## 🛠 Installation

1. Install dependencies:
```bash
pip install google-genai opencv-python pyaudio pillow mss numpy
```

2. Set up your Gemini API key:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

## 🎯 Usage Examples

### Autonomous Flight Status Check
```bash
python app.py --task "check flight status AA1123"
```

### Reservation Information
```bash
python app.py --task "get reservation info ABC123"
```

### Delta Airlines Flight Check
```bash
python app.py --task "check Delta flight DL2456 status"
```

### Custom Instructions
```bash
python app.py --task "call United about baggage" --instructions "Be polite and ask for supervisor if needed"
```

### Interactive Mode
```bash
python app.py --mode none
```
Then type: `auto: check my American flight AA1123`

## 🎮 Demo Mode

Run the demo to see how the agent works:

```bash
python demo.py
```

This will simulate the agent calling an airline and navigating through the IVR system with actual DTMF tones.

## 📞 Supported Airlines

- **American Airlines**: 1-800-433-7300 (AA)
- **Delta Airlines**: 1-800-221-1212 (DL)  
- **United Airlines**: 1-800-864-8331 (UA)
- **Southwest Airlines**: 1-800-435-9792 (WN)
- **JetBlue Airways**: 1-800-538-2583 (B6)

## 🧠 How It Works

1. **Task Parsing**: The system parses your request to extract:
   - Task type (flight status, reservation, etc.)
   - Airline information
   - Flight numbers or confirmation codes

2. **AI Instructions**: Creates specific instructions for the AI agent based on your task

3. **Function Tools**: The AI can use these tools:
   - `press_number()`: Generate DTMF tones for menu navigation
   - `provide_flight_info()`: Speak flight numbers, confirmation codes
   - `wait_for_menu()`: Pause to listen to menu options
   - `log_call_progress()`: Keep you informed of progress
   - `extract_flight_details()`: Save flight information

4. **Autonomous Navigation**: The AI listens to menu options and makes appropriate choices

5. **Result Extraction**: Captures and saves the information you requested

## 🔧 Interactive Commands

While the agent is running:

- `status`: View current call progress
- `auto: <task>`: Give the agent a new autonomous task
- `q`: Quit the application

## 🎛 Configuration Options

- `--mode`: Video mode (`camera`, `screen`, `none`)
- `--task`: Specific task for autonomous operation
- `--instructions`: Additional instructions for the AI
- `--log-level`: Logging detail level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--log-file`: Save detailed logs to a specific file

## 📝 Output Files

The agent saves comprehensive information:
- **Result Files**: `flight_result_YYYYMMDD_HHMMSS.txt` - Flight info with call statistics
- **Log Files**: `ai_agent_log_YYYYMMDD_HHMMSS.log` - Detailed technical logs (auto-generated for tasks)
- **Custom Logs**: Your specified filename when using `--log-file`

## 📊 Comprehensive Logging

The system now includes detailed logging to help you understand what's happening:

### Real-time Console Logs
- **📞 DTMF**: Shows when keys are pressed and why
- **ℹ️ INFO**: Information being provided to the system
- **⏳ WAIT**: When the agent is waiting for menus
- **📊 STATUS**: Call progress updates
- **✅ RESULT**: Flight information received
- **❌ ERROR**: Any errors that occur
- **🔗 SESSION**: Connection status
- **🔊 AUDIO**: Audio system status

### Interactive Commands
While the agent is running, you can use these commands:

- `status` - Show current call statistics and recent logs
- `logs` - Display the complete call log
- `debug` - Toggle detailed debug logging on/off
- `help` - Show all available commands
- `auto: <task>` - Give the agent a new autonomous task
- `q` - Quit the application

### Log Levels
Control the amount of detail in logs:

```bash
# Basic info only
python app.py --task "check AA1123" --log-level INFO

# Detailed debugging
python app.py --task "check AA1123" --log-level DEBUG

# Save logs to file
python app.py --task "check AA1123" --log-file my_call.log

# Combine options
python app.py --task "check AA1123" --log-level DEBUG --log-file debug_call.log
```

### Log Files
- **Automatic**: For autonomous tasks, logs are auto-saved to timestamped files
- **Manual**: Use `--log-file filename.log` to specify a custom log file
- **Content**: Includes detailed function calls, timing, audio stats, and full call history

## 🚨 Important Notes

1. **Real Phone Calls**: This system generates real DTMF tones - be careful not to accidentally call numbers
2. **API Costs**: Uses Google Gemini Live API which may have usage costs
3. **Audio Quality**: Ensure good microphone/speaker setup for best results
4. **Testing**: Use `demo.py` first to understand how the system works

## 🔍 Task Examples

### Flight Status Queries
- "check flight status AA1123"
- "flight status Delta 2456"  
- "is United flight UA789 on time"

### Reservation Management
- "get reservation info ABC123"
- "check booking confirmation XYZ789"
- "find my American reservation"

### General Inquiries
- "call Southwest about baggage policy"
- "check JetBlue flight change options"
- "United customer service for refund"

## 🛟 Troubleshooting

### Audio Issues
- Check microphone/speaker permissions
- Verify PyAudio installation: `pip install pyaudio`

### API Issues  
- Verify `GEMINI_API_KEY` is set correctly
- Check API quota and billing

### DTMF Issues
- Ensure speakers are working
- Adjust volume in `dtmf_generator.py`

## 🔮 Future Enhancements

- [ ] More airline support
- [ ] Hotel and rental car bookings
- [ ] Integration with calendar apps
- [ ] Voice recognition improvements
- [ ] Multi-language support
- [ ] GUI interface

## ⚠️ Disclaimer

This is a proof-of-concept system. Use responsibly and ensure you have permission before making automated calls to customer service lines. Some airlines may have policies against automated systems.

## 📄 License

This project is for educational and research purposes. Please comply with all applicable laws and terms of service.
