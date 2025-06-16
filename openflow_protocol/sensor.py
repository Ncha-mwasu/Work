import socket
import time
#import a

# Sensor configuration
SENSOR_IP = "10.0.0.1"    # Mininet h1 IP
DEST_IP = "10.0.0.2"      # Mininet h2 IP
UDP_PORT = 12345
SENSOR_DATA = "temperature:25.5"

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send data
while True:
    sock.sendto(SENSOR_DATA.encode(), (DEST_IP, UDP_PORT))
    print(f"Sent: {SENSOR_DATA} to {DEST_IP}:{UDP_PORT}")
    time.sleep(2)

sock.close()


NODE_IP_BASE = "192.168.1."
NODE_IPS = {i: f"{NODE_IP_BASE}{i+2}" for i in range(NB_NODES)}  # 192.168.1.2 to 192.168.1.31
CONTROLLER_IPS = {
    BSID: "192.168.1.1",  # Base station
    SUBCONT0: f"{NODE_IP_BASE}32",  # 192.168.1.32
    SUBCONT1: f"{NODE_IP_BASE}33"  # 192.168.1.33
}