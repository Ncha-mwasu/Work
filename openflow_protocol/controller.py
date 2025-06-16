# controller.py
import socket
import struct

from pyof.v0x01.controller2switch.flow_mod import FlowMod
from pyof.v0x01.common.flow_match import Match
from pyof.v0x01.common.action import ActionOutput
from pyof.v0x01.common.phy_port import Port
from pyof.utils import unpack

# Controller settings
CONTROLLER_IP = "0.0.0.0"  # Listen on all interfaces
CONTROLLER_PORT = 6633

# Create TCP server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((CONTROLLER_IP, CONTROLLER_PORT))
sock.listen(1)
print(f"Controller listening on {CONTROLLER_IP}:{CONTROLLER_PORT}")

# Accept switch connection
(conn, addr) = sock.accept()
print(f"Switch connected: {addr}")

# def handle_packet_in(binary_message):
#     """Parse Packet In messages and extract sensor data"""
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

# def install_flow_rule(conn):
#     """Install a flow rule to forward sensor packets"""
#     flow_mod = FlowMod()
#     flow_mod.match = Match(nw_src="192.168.56.101")  # Sensor IP
#     flow_mod.command = 0  # OFPFC_ADD
#     flow_mod.actions.append(ActionOutput(port=Port.OFPP_2))  # Forward to port 2
#     conn.send(flow_mod.pack())
#     print("Flow rule installed")

# # Main loop
# while True:
#     data = conn.recv(1024)
#     if not data:
#         break
#     if handle_packet_in(data):
#         install_flow_rule(conn)
#     print(f'Received data: {data}')

# conn.close()
# sock.close()