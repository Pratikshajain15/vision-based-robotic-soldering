/**
 * ESP32 Controller for Soldering Robot Power & Heating
 * 
 * Hardware Connections:
 * - ESP32 pin 18 -> Relay Module IN1 (24V SMPS AC control)
 * - ESP32 pin 19 -> Relay Module IN2 (Soldering Iron Heater control)
 * - ESP32 GND    -> Relay Module GND
 * - ESP32 5V/VBUS-> Relay Module VCC (For 5V relay trigger)
 * 
 * USB connection to Raspberry Pi 5 provides communication & power.
 */

// Pin Definitions
const int RELAY_SMPS_PIN = 18;   // Channel 1 Relay: Controls SMPS AC Input
const int RELAY_HEATER_PIN = 19; // Channel 2 Relay: Controls Soldering Iron Heater (24V DC side)

// Relay Logic Configuration
// Most standard 5V multi-channel relay modules are Active LOW (low signal turns relay ON).
// Set this to false if your relay board is Active HIGH.
const bool RELAY_ACTIVE_LOW = true;

// Helper variables for non-blocking reports
unsigned long lastStatusTime = 0;
const unsigned long STATUS_INTERVAL = 2000; // Send status every 2 seconds

// State variables
bool smpsState = false;
bool heaterState = false;

// Function to set relay state based on configuration
void setRelay(int pin, bool turnOn) {
  if (RELAY_ACTIVE_LOW) {
    digitalWrite(pin, turnOn ? LOW : HIGH);
  } else {
    digitalWrite(pin, turnOn ? HIGH : LOW);
  }
}

void setup() {
  // Initialize Serial communication over USB at 115200 baud
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for native USB)
  }

  // Configure relay pins as output and set initial state to OFF
  pinMode(RELAY_SMPS_PIN, OUTPUT);
  pinMode(RELAY_HEATER_PIN, OUTPUT);
  
  setRelay(RELAY_SMPS_PIN, false);
  setRelay(RELAY_HEATER_PIN, false);

  Serial.println("ESP32_CONTROLLER_READY");
}

void loop() {
  // Check for incoming commands from Raspberry Pi
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove whitespace and newline characters

    if (command == "PING") {
      Serial.println("PONG_ESP32");
    } 
    else if (command == "SYSTEM_ON") {
      // Turn on 24V SMPS and heat the soldering iron
      smpsState = true;
      heaterState = true;
      setRelay(RELAY_SMPS_PIN, true);
      delay(100); // Small delay to avoid inrush current spike
      setRelay(RELAY_HEATER_PIN, true);
      Serial.println("SYSTEM_OK");
    } 
    else if (command == "SYSTEM_OFF") {
      // Turn off heating first
      heaterState = false;
      setRelay(RELAY_HEATER_PIN, false);
      delay(500); // Give heater circuit a moment to switch off
      // Turn off SMPS
      smpsState = false;
      setRelay(RELAY_SMPS_PIN, false);
      Serial.println("SYSTEM_SHUTDOWN");
    } 
    else if (command == "HEATER_ON") {
      heaterState = true;
      setRelay(RELAY_HEATER_PIN, true);
      Serial.println("HEATER_OK");
    } 
    else if (command == "HEATER_OFF") {
      heaterState = false;
      setRelay(RELAY_HEATER_PIN, false);
      Serial.println("HEATER_OFF_OK");
    } 
    else if (command == "STATUS") {
      sendStatus();
    }
    else {
      Serial.print("ERROR_UNKNOWN_COMMAND: ");
      Serial.println(command);
    }
  }

  // Periodically send status telemetry to Pi
  if (millis() - lastStatusTime >= STATUS_INTERVAL) {
    lastStatusTime = millis();
    sendStatus();
  }
}

void sendStatus() {
  Serial.print("STATUS|SMPS:");
  Serial.print(smpsState ? "ON" : "OFF");
  Serial.print("|HEATER:");
  Serial.println(heaterState ? "ON" : "OFF");
}
