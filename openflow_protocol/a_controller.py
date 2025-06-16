import socket
import struct
# from pyof.v0x01.foundation.basic_types import UBInt8, UBInt32
from pyof.v0x01.controller2switch.flow_mod import FlowMod
from pyof.v0x01.common.flow_match import Match
from pyof.v0x01.common.action import ActionOutput
from pyof.v0x01.common.phy_port import Port
#from pyof.v0x01.foundation.basic_types import UBInt8, UBInt32
from pyof.utils import unpack
import sensor

# Controller settings
CONTROLLER_IP = "0.0.0.0"  # Listen on all interfaces
CONTROLLER_PORT = 6633

# Create TCP server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)  # Timeout to avoid hanging
sock.bind((CONTROLLER_IP, CONTROLLER_PORT))
sock.listen(1)
print(f"Controller listening on {CONTROLLER_IP}:{CONTROLLER_PORT}")

try:
    # Accept switch connection
    conn, addr = sock.accept()
    print(f"Switch connected: {addr}")
except socket.timeout:
    print("Error: No switch connected within 10 seconds")
    sock.close()
    exit(1)

# def send_flow_mod(conn):
#     """Send a FlowMod message to forward UDP packets"""
#     flow_mod = FlowMod()
#     match = Match()
#     match.nw_src = "10.0.0.1"  # Sensor IP (e.g., Mininet h1)
#     match.nw_dst = "10.0.0.2"  # Destination IP (e.g., Mininet h2)
#     match.nw_proto = 17  # UDP protocol
#     match.tp_dst = 12345  # UDP destination port
#     flow_mod.match = match
#     flow_mod.command = 0  # OFPFC_ADD (add flow rule)
#     flow_mod.actions.append(ActionOutput(port=Port.OFPP_2))  # Forward to port 2
#     conn.send(flow_mod.pack())
#     print("FlowMod message sent")

# def handle_packet_in(binary_message):
#     """Handle Packet In messages (optional, for sensor data)"""
#     try:
#         msg = unpack(binary_message)
#         if msg.header.message_type == 10:  # OFPT_PACKET_IN
#             print("Received Packet In")
#             sensor_data = msg.data.value.decode('utf-8', errors='ignore')
#             print(f"Sensor Data: {sensor_data}")
#             return True
#     except Exception as e:
#         print(f"Error parsing message: {e}")
#     return False

# # Send FlowMod immediately after connection
# send_flow_mod(conn)

# # Main loop to handle incoming messages (optional)
# while True:
#     try:
#         data = conn.recv(1024)
#         if not data:
#             print("Switch disconnected")
#             break
#         handle_packet_in(data)
#     except socket.timeout:
#         print("No data received, continuing...")
#         continue
#     except Exception as e:
#         print(f"Error: {e}")
#         break

# conn.close()
# sock.close()
