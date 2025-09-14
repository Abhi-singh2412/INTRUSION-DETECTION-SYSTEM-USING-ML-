import pandas as pd
import joblib
import sys
import os
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, Raw
import time
from collections import defaultdict
from urllib.parse import urlparse
from datetime import datetime
import requests # New import for API calls
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("VIRUSTOTAL_API_KEY")
if api_key is None:
    print("API key not found. Please set the 'VIRUSTOTAL_API_KEY' in your .env file.")
else:
    print("API key loaded successfully.")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the project root)
project_root = os.path.dirname(script_dir)
# Add the project root to the system path to allow importing modules
sys.path.append(project_root)


from src.data_preprocessing import load_and_preprocess


sessions = defaultdict(lambda: {'packets': 0, 'start_time': time.time(), 'failed_logins': 0, 'last_packet_time': time.time()})
ip_reputation_cache = {} # Cache to store recent IP scores to reduce API calls

def get_ip_reputation(ip_address):
    if ip_address in ip_reputation_cache:
        return ip_reputation_cache[ip_address]
        
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip_address}"
    headers = {
        "x-apikey": api_key,
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an exception for bad status codes
        data = response.json()
        
        # VirusTotal's reputation is based on "harmless," "malicious," "suspicious," and "undetected"
        # We can map this to a numeric score.
        last_analysis_stats = data['data']['attributes']['last_analysis_stats']
        malicious_score = last_analysis_stats.get('malicious', 0)
        harmless_score = last_analysis_stats.get('harmless', 0)
        
        # Simple scoring logic: more malicious detections = higher score
        total_votes = malicious_score + harmless_score
        if total_votes > 0:
            reputation = malicious_score / total_votes
        else:
            reputation = 0.5 # Neutral score if no data is available
        
        ip_reputation_cache[ip_address] = reputation
        return reputation

    except requests.exceptions.RequestException as e:
        print(f"Error fetching IP reputation for {ip_address}: {e}")
        return 0.5 # Return a neutral score if the API call fails
    except KeyError:
        # IP not found in VirusTotal data
        return 0.5

def get_browser_type(packet):
    #Parses the packet's raw data to find and extract the browser from the User-Agent header.

    # Check if the packet has a Raw payload and is a TCP packet (typical for HTTP)
    if packet.haslayer(Raw) and TCP in packet:
        try:
            # Decode the payload
            payload = packet[Raw].load.decode('utf-8', errors='ignore')
            
            # Check for common browser keywords in the User-Agent header
            if "User-Agent" in payload:
                user_agent_line = payload.split("User-Agent:")[1].split('\r\n')[0]
                if "Chrome" in user_agent_line:
                    return "Chrome"
                elif "Firefox" in user_agent_line:
                    return "Firefox"
                elif "Safari" in user_agent_line and "Chrome" not in user_agent_line:
                    return "Safari"
                elif "Edge" in user_agent_line:
                    return "Edge"
                elif "Opera" in user_agent_line:
                    return "Opera"
        except (UnicodeDecodeError, IndexError, AttributeError):
            pass
    return "Unknown"

def feature_engineering(data):
    """
    Performs feature engineering to create new, more valuable features from the raw data.
    """
    data['failed_login_rate'] = data['failed_logins'] / (data['login_attempts'] + 1e-6)
    data['session_speed'] = data['session_duration'] / (data['network_packet_size'] + 1e-6)
    data['protocol_encryption_combo'] = data['protocol_type'] + '_' + data['encryption_used'].astype(str)
    
    return data

def process_packet(packet):
    """
    This function processes each live packet, extracts features, and uses the model
    to make a real-time prediction.
    """
    if IP in packet:
        try:
            # Create a unique session ID from the source and destination IP/port
            if TCP in packet:
                session_id = f"{packet[IP].src}:{packet[TCP].sport}-{packet[IP].dst}:{packet[TCP].dport}"
                protocol = 'TCP'
                # Check for encryption (TLS/SSL Handshake)
                if packet.haslayer(Raw) and b'\x16\x03' in packet[Raw].load:
                    encryption_used = 'True'
                else:
                    encryption_used = 'False'
            elif UDP in packet:
                session_id = f"{packet[IP].src}:{packet[UDP].sport}-{packet[IP].dst}:{packet[UDP].dport}"
                protocol = 'UDP'
                encryption_used = 'False'
            else:
                session_id = f"{packet[IP].src}-{packet[IP].dst}"
                protocol = 'Unknown'
                encryption_used = 'False'

            # Update session stats
            sessions[session_id]['packets'] += 1
            sessions[session_id]['session_duration'] = time.time() - sessions[session_id]['start_time']
            
            #  FEATURE EXTRACTION FROM PACKET 
            data = {
                'session_id': [session_id],
                'network_packet_size': [len(packet)],
                'protocol_type': [protocol],
                'login_attempts': [sessions[session_id]['packets']],
                'session_duration': [sessions[session_id]['session_duration']],
                'encryption_used': [encryption_used], 
                'ip_reputation_score': [get_ip_reputation(packet[IP].src)],
                'failed_logins': [0], # Placeholder, requires deeper inspection
                'browser_type': [get_browser_type(packet)], 
                'unusual_time_access': [1 if datetime.now().hour > 22 or datetime.now().hour < 6 else 0]
            }
            
            # Convert to DataFrame
            sample_data = pd.DataFrame(data)

            # PREPROCESSING AND PREDICTION 
            # Apply feature engineering to the sample data
            engineered_sample_data = feature_engineering(sample_data)

            # Preprocess the sample data using the loaded preprocessor
            processed_sample = preprocessor.transform(engineered_sample_data)
            
            # Make a prediction
            prediction = rf_model.predict(processed_sample)
            
            # Print the result in real-time
            print("\n" + "="*50)
            print(f"[{time.strftime('%H:%M:%S')}] Packet from {packet[IP].src} to {packet[IP].dst}")
            print(f"Protocol: {protocol}, Size: {len(packet)} bytes")
            
            if prediction[0] == 1:
                print("\n\033[91m!!! ATTACK DETECTED !!!\033[0m")
            else:
                print("\n\033[92m--- Normal Traffic Detected ---\033[0m")
            print("="*50)

        except Exception as e:
            # Handle packets that don't fit the expected format
            print(f"Could not process packet: {e}")

if __name__ == "__main__":
    # Define file paths
    model_dir = os.path.join(project_root, 'models')
    
    # Load the trained model and preprocessor
    try:
        rf_model = joblib.load(os.path.join(model_dir, 'ids_model.pkl'))
        preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
        print("Model and preprocessor loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or preprocessor files not found. Please run src/train_random_forest.py first.")
        sys.exit(1)

    print("\n--- Starting Live Network Analysis (Press Ctrl+C to stop) ---")
    
    sniff(prn=process_packet, iface="Wi-Fi")
