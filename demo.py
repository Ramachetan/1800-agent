"""
Demo script to simulate the AI phone agent interacting with airline IVR systems
This demonstrates how the agent would navigate through typical airline menus
"""

import asyncio
import time
from dtmf_generator import DTMFGenerator

class MockAirlineIVR:
    """Mock airline IVR system for testing"""
    
    def __init__(self, airline_name="American Airlines"):
        self.airline_name = airline_name
        self.state = "greeting"
        self.dtmf = DTMFGenerator()
        
    async def play_greeting(self):
        print(f"\n🔊 {self.airline_name} IVR System:")
        print("Thank you for calling American Airlines.")
        print("Para español, presiona dos.")
        print("For flight information, press 1")
        print("For existing reservations, press 2") 
        print("For customer service, press 0")
        print("Please make your selection now.")
        
        # Wait for DTMF input simulation
        await asyncio.sleep(2)
        
    async def handle_flight_info_menu(self):
        print(f"\n🔊 Flight Information Menu:")
        print("You have selected flight information.")
        print("For departures, press 1")
        print("For arrivals, press 2")
        print("For flight status by flight number, press 3")
        print("To return to the main menu, press 9")
        
        await asyncio.sleep(2)
        
    async def handle_flight_status_input(self):
        print(f"\n🔊 Flight Status:")
        print("Please enter your flight number followed by the pound key.")
        print("For example, for flight 1234, press 1-2-3-4-#")
        
        await asyncio.sleep(3)
        
    async def provide_flight_status(self, flight_number="1123"):
        print(f"\n🔊 Flight Status for AA{flight_number}:")
        print(f"American Airlines flight {flight_number}")
        print("Departure: New York JFK at 2:30 PM")
        print("Arrival: Los Angeles LAX at 5:45 PM Pacific Time")
        print("Status: ON TIME")
        print("Gate: A12")
        print("Thank you for calling American Airlines.")
        
    def close(self):
        self.dtmf.close()

class AIAgentDemo:
    """Simulates the AI agent navigating the IVR"""
    
    def __init__(self, task="check flight status AA1123"):
        self.task = task
        self.ivr = MockAirlineIVR()
        self.dtmf = DTMFGenerator()
        
    async def simulate_agent_actions(self):
        print("=" * 60)
        print("🤖 AI PHONE AGENT DEMO")
        print("=" * 60)
        print(f"Task: {self.task}")
        print("Agent is now 'calling' American Airlines...")
        
        # Step 1: Listen to greeting
        await self.ivr.play_greeting()
        
        # Step 2: Agent decides to press 1 for flight information
        print(f"\n🤖 AI Agent: I need flight information, pressing 1...")
        await self.dtmf.play_tone("1")
        await asyncio.sleep(0.5)
        
        # Step 3: Navigate flight info menu
        await self.ivr.handle_flight_info_menu()
        
        # Step 4: Agent presses 3 for flight status by flight number
        print(f"\n🤖 AI Agent: I need flight status by number, pressing 3...")
        await self.dtmf.play_tone("3")
        await asyncio.sleep(0.5)
        
        # Step 5: IVR asks for flight number
        await self.ivr.handle_flight_status_input()
        
        # Step 6: Agent enters flight number
        flight_number = "1123"
        print(f"\n🤖 AI Agent: Entering flight number {flight_number}...")
        await self.dtmf.play_sequence(flight_number + "#", pause_between=0.3)
        
        # Step 7: IVR provides flight status
        await asyncio.sleep(1)
        await self.ivr.provide_flight_status(flight_number)
        
        # Step 8: Agent completes task
        print(f"\n✅ AI Agent: Task completed! Flight information retrieved.")
        print(f"📋 Result: AA1123 is ON TIME, departing JFK at 2:30 PM")
        
    async def run_demo(self):
        try:
            await self.simulate_agent_actions()
        finally:
            self.ivr.close()
            self.dtmf.close()

async def run_multiple_scenarios():
    """Run different scenarios to show various capabilities"""
    
    scenarios = [
        "check flight status AA1123",
        "get reservation info for confirmation ABC123", 
        "check baggage status for United flight"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*20} SCENARIO {i} {'='*20}")
        demo = AIAgentDemo(scenario)
        if i == 1:  # Only run full demo for first scenario
            await demo.run_demo()
        else:
            print(f"Task: {scenario}")
            print("🤖 AI Agent would navigate appropriate menus...")
            print("✅ Task completed successfully")
        
        demo.ivr.close()
        demo.dtmf.close()
        
        if i < len(scenarios):
            await asyncio.sleep(2)

if __name__ == "__main__":
    print("🚀 Starting AI Phone Agent Demo...")
    print("This demonstrates how the agent navigates airline IVR systems")
    print("You'll hear actual DTMF tones being generated!")
    
    # Run the demo
    asyncio.run(run_multiple_scenarios())
