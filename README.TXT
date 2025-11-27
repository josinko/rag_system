New-NetFirewallRule -DisplayName "RAG API Server" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow

This PowerShell command creates a Windows Firewall rule to allow network traffic. Let me break it down:
Command Breakdown:
New-NetFirewallRule

Creates a new firewall rule in Windows

-DisplayName "RAG API Server"
The name you'll see in Windows Firewall settings
Just a label for humans to identify this rule

-Direction Inbound
Inbound = Traffic coming INTO your computer from the network
(Outbound would be traffic going OUT from your computer)

-LocalPort 5000
The port number on your computer that will accept connections
Your Flask API runs on port 5000

-Protocol TCP
Use TCP protocol (Transmission Control Protocol)
HTTP/REST APIs use TCP

-Action Allow
Allow the traffic (don't block it)
(Other option would be "Block")

*What It Does in Simple Terms:

Without this rule:
Your Flask API runs on port 5000
Windows Firewall blocks all incoming connections from other computers
Only your own computer can access http://localhost:5000
Computer B CANNOT connect to your API

With this rule:
Windows Firewall allows incoming connections on port 5000
Other computers on the network can now connect
Computer B CAN connect to http://192.168.1.100:5000

Real-World Analogy:
Think of your computer as a building:

Port 5000 = Door number 5000
Windows Firewall = Security guard at the entrance
Without the rule = Guard blocks everyone trying to enter door 5000
With the rule = Guard allows people to enter through door 5000

How to Run It:
1. Open PowerShell as Administrator:
-Press Win + X
-Click "Windows PowerShell (Admin)" or "Terminal (Admin)"

2. Paste the command:
powershell
New-NetFirewallRule -DisplayName "RAG API Server" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
```

**3. Press Enter**

You should see output like:
```
Name                  : {Generated-GUID}
DisplayName           : RAG API Server
Description           : 
DisplayGroup          : 
Group                 : 
Enabled               : True
Profile               : Any
Platform              : {}
Direction             : Inbound
Action                : Allow
EdgeTraversalPolicy   : Block
...

Verify the Rule:
Option 1: PowerShell
powershell Get-NetFirewallRule -DisplayName "RAG API Server"

Option 2: Windows Firewall GUI
1-Press Win + R
2-Type: wf.msc
3-Press Enter
4-Click "Inbound Rules" on left
5-Find "RAG API Server" in the list
6-You'll see it's enabled and allows port 5000

To Remove the Rule Later:
powershell Remove-NetFirewallRule -DisplayName "RAG API Server"

Why This Is Needed:
By default, Windows Firewall blocks all incoming connections for security. This is good! But when you want to run a server (like your RAG API) that other computers need to access, you must explicitly allow it.

Think of it as:
-Your Flask app says: "I'm ready to accept connections on port 5000!"
-Windows Firewall says: "Not so fast! I'm blocking that port for security."
-This command tells Firewall: "It's okay, allow port 5000 for the RAG API."



RAM (Computer Memory)
┌─────────────────────────────────────┐
│ Process: python.exe (PID 5432)      │
│ ┌─────────────────────────────────┐ │
│ │ Python Interpreter              │ │
│ │ ┌─────────────────────────────┐ │ │
│ │ │ Your rag_api.py code        │ │ │
│ │ │ - Flask app                 │ │ │
│ │ │ - qa_chain (loaded)         │ │ │
│ │ │ - vectorstore (loaded)      │ │ │
│ │ │ - Listening on port 5000    │ │ │
│ │ └─────────────────────────────┘ │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘


When Flask runs app.run(host='0.0.0.0', port=5000):
What happens behind the scenes:

1-Flask asks Windows: "Can I use port 5000?"
2-Windows checks: Is port 5000 already in use?
-If YES → Error: "Address already in use"
-If NO → "OK, port 5000 is yours"
3-Windows creates a socket (network endpoint)
4-Socket binds to port 5000
5-Flask starts listening on that socket
6-Windows routes any traffic arriving at port 5000 to your Flask process

Network Traffic Flow:
┌─────────────────────────────────────────────────────┐
│ Internet/Network                                    │
└────────────────┬────────────────────────────────────┘
                 │
                 │ Request: http://192.168.1.100:5000/api/ask
                 ↓
┌─────────────────────────────────────────────────────┐
│ Network Interface (Your computer's network card)    │
└────────────────┬────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────┐
│ Windows Operating System                            │
│ "This is for port 5000... let me check who owns it" │
└────────────────┬────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────┐
│ Your Flask Process (python.exe PID 5432)           │
│ Listening on port 5000                              │
│ Receives: POST /api/ask {"question": "..."}        │
│ Processes request                                   │
│ Returns: {"answer": "..."}                         │
└─────────────────────────────────────────────────────┘


You → Terminal → python.exe → rag_api.py → Flask → Port 5000
         ↑
    (visible, controllable)


Windows Boot → services.exe → Your Service → python.exe → rag_api.py → Flask → Port 5000
                                                ↑
                                        (hidden, background)



Important: Whether running from terminal or as a service, the port binding is identical:
python app.run(host='0.0.0.0', port=5000)

What this means:
1-Process starts (terminal or service - doesn't matter)
2-Flask asks Windows: "Give me port 5000"
3-Windows creates socket bound to port 5000
4-Process stays in memory and listens
5-Any network request to port 5000 goes to this process
6-Process responds through the same port

The code lives in memory either way:
-Terminal: In user session memory
-Service: In system session memory
-But port 5000 is occupied the same way in both cases   


Let's trace a request: http://192.168.1.100:5000/api/ask
Step-by-step:
1-Client (WPF app) sends HTTP request → Network card
2-Network card → Windows network stack
3-Windows checks: "Who owns port 5000?"
4-Windows finds: PID 5432 (your Flask process)
5-Windows wakes up your process (if sleeping)
6-Flask receives the request:
python   @app.route('/api/ask', methods=['POST'])
   def ask_question():
       # This function executes
7-Your code processes it:
-Reads question from request
-Queries vector database (in memory)
-Calls Ollama LLM
-Gets answer
8-Flask sends HTTP response back through port 5000
9-Windows routes response back to client
10-Client receives answer

All of this happens in memory - the process is already loaded and ready!



# Instance 1
app.run(host='0.0.0.0', port=5000)

# Instance 2
app.run(host='0.0.0.0', port=5001)

# Instance 3
app.run(host='0.0.0.0', port=5002)
```

---

## Part 8: Memory Persistence

### **Why does the service need to stay in memory?**

**Your RAG system has loaded:**
- Vector database (ChromaDB) - could be GBs of data
- LLM connection (Ollama)
- Embeddings model
- Document chunks
- All the infrastructure

**If the process stops:**
- All that memory is freed
- Everything is lost
- Next request would fail (nothing listening on port 5000)
- Would need to reload everything (minutes!)

**By keeping it in memory:**
- Instant responses
- No reload time
- Always ready

---

## Part 9: Process Lifecycle Comparison

### **Terminal Process:**
```
You run: python rag_api.py
   ↓
Process starts in memory
   ↓
Flask binds to port 5000
   ↓
System loads (minutes)
   ↓
Ready to serve requests ✅
   ↓
You close terminal
   ↓
Process dies ⚰️
   ↓
Port 5000 freed
   ↓
Next request fails ❌
```

### **Windows Service:**
```
Computer boots
   ↓
Service auto-starts
   ↓
Process starts in memory
   ↓
Flask binds to port 5000
   ↓
System loads (minutes)
   ↓
Ready to serve requests ✅
   ↓
You log out (service keeps running)
   ↓
Still ready to serve requests ✅
   ↓
Computer shuts down
   ↓
Service stops gracefully
   ↓
Process dies ⚰️



Summary Table:
Aspect  Terminal    Windows Service Port Behavior
Runs in Console window  Background  Same
Visible Yes No  N/A
Survives logout No  Yes Same
Auto-start on boot  No  Yes (if configured) Same
Memory location User session    System session  Same RAM
Port binding    Same    Same    Identical!
Process type    Interactive Service Both are processes
Can coexist?    No (same port)  No (same port)  Only one can own a port

Key insight: Terminal vs. Service is about process lifecycle management, not about how the code runs or how ports work. The actual Flask app, port binding, and request handling are 100% identical in both cases!