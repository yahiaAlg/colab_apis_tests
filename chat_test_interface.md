```python
from pprint import pprint
import ipywidgets as widgets
from IPython.display import display, HTML
import requests
import json
from datetime import datetime
import io

# Global variables
base_url = None
file_upload = None
chat_input = None
chat_history = []
send_button = None
clear_button = None
output_area = None
system_output = None
status = None
context_info = None


def create_widgets():
    global file_upload, chat_input, send_button, clear_button, output_area
    global system_output, status, context_info

    # File upload widget
    file_upload = widgets.FileUpload(
        description="Upload Document", accept=".txt,.pdf,.md", multiple=False
    )

    # Chat input
    chat_input = widgets.Textarea(
        placeholder="Type your message here...",
        layout=widgets.Layout(width="90%", height="100px"),
    )

    # Buttons
    send_button = widgets.Button(
        description="Send", button_style="primary", icon="paper-plane"
    )

    clear_button = widgets.Button(
        description="Clear Chat", button_style="warning", icon="trash"
    )

    # Output areas
    output_area = widgets.Output(
        layout={
            "border": "1px solid #ddd",
            "padding": "10px",
            "margin": "10px 0px",
            "max-height": "400px",
            "overflow-y": "auto",
        }
    )

    system_output = widgets.Output(
        layout={"border": "1px solid #ddd", "padding": "10px"}
    )

    # Context info area
    context_info = widgets.HTML(
        value="<b>Context ID:</b> None", layout={"margin": "10px 0px"}
    )

    # Status indicator
    status = widgets.HTML(value="<b>Status:</b> Ready")

    # Connect callbacks
    send_button.on_click(handle_chat)
    clear_button.on_click(clear_chat)
    file_upload.observe(handle_file_upload, names="value")


def clear_chat(b):
    global chat_history
    chat_history = []
    output_area.clear_output()
    context_info.value = "<b>Context ID:</b> None"
    with output_area:
        print("Chat history cleared.")


def handle_file_upload(change):
    if not file_upload.value:
        return

    try:
        status.value = "<b>Status:</b> Uploading document..."
        file_info = list(file_upload.value.values())[0]
        files = {"file": (file_info.name, file_info.content)}

        response = requests.post(f"{base_url}/api/document/upload", files=files)
        result = response.json()

        context_info.value = f"<b>Context ID:</b> {result['context_id']}"

        with output_area:
            print(f"Document uploaded successfully: {file_info.name}")
            print(f"Document type: {result['metadata']['type']}")
            print(f"Size: {result['metadata']['size']} bytes")
            print("\nPreview of processed content:")
            print("-" * 50)
            print(
                result["content"][:500] + "..."
                if len(result["content"]) > 500
                else result["content"]
            )
            print("-" * 50)

        status.value = "<b>Status:</b> Ready"

    except Exception as e:
        with output_area:
            print(f"Error uploading document: {str(e)}")
        status.value = "<b>Status:</b> Error occurred"


def handle_chat(b):
    if not chat_input.value.strip():
        return

    try:
        status.value = "<b>Status:</b> Processing..."

        # Get context_id if available
        context_id = None
        if "None" not in context_info.value:
            context_id = context_info.value.split("Context ID:</b> ")[1]

        # Add user message to chat history
        chat_history.append({"role": "user", "content": chat_input.value.strip()})

        # Prepare request
        payload = {"messages": chat_history, "context_id": context_id}

        # Send request
        response = requests.post(f"{base_url}/api/chat", json=payload)
        result = response.json()

        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": result["response"]})

        # Update display
        output_area.clear_output()
        with output_area:
            for msg in chat_history:
                if msg["role"] == "user":
                    display(
                        HTML(
                            f'<div style="margin: 5px; padding: 10px; background-color: #e3f2fd; border-radius: 10px;"><b>You:</b> {msg["content"]}</div>'
                        )
                    )
                else:
                    display(
                        HTML(
                            f'<div style="margin: 5px; padding: 10px; background-color: #f5f5f5; border-radius: 10px;"><b>Assistant:</b> {msg["content"]}</div>'
                        )
                    )

        # Clear input
        chat_input.value = ""
        status.value = "<b>Status:</b> Ready"

    except Exception as e:
        with output_area:
            print(f"Error: {str(e)}")
        status.value = "<b>Status:</b> Error occurred"


def display_widgets():
    # Create tabs
    chat_tab = widgets.VBox(
        [
            widgets.HTML(value="<h3>Chat Interface</h3>"),
            chat_input,
            widgets.HBox([send_button, clear_button]),
            context_info,
            output_area,
        ]
    )

    upload_tab = widgets.VBox(
        [widgets.HTML(value="<h3>Document Upload</h3>"), file_upload, system_output]
    )

    tabs = widgets.Tab(children=[chat_tab, upload_tab])
    tabs.set_title(0, "Chat")
    tabs.set_title(1, "Upload")

    display(status)
    display(tabs)


def initialize_client(api_url):
    global base_url
    base_url = api_url
    create_widgets()
    display_widgets()
