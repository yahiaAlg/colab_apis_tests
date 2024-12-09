```python
from pprint import pprint
import ipywidgets as widgets

from IPython.display import display, HTML

import requests

import json

from datetime import datetime


# Global variables

collection_id = None

base_url = None

file_upload = None

question_input = None

upload_button = None
check_button = None

ask_button = None
upload_output = None
system_output = None

qa_output = None

status = None



def create_widgets():

    global file_upload, question_input, upload_button, ask_button, upload_output, qa_output, system_output , status, check_button


    # File upload widget

    file_upload = widgets.FileUpload(

        description="Upload Document", accept=".pdf,.txt", multiple=False
    )


    # Question input

    question_input = widgets.Text(

        description="Question:",

        placeholder="Enter your question here...",

        disabled=True,

        layout=widgets.Layout(width="80%"),
    )


    # Buttons

    upload_button = widgets.Button(description="Upload")
    check_button = widgets.Button(description="check")

    ask_button = widgets.Button(description="Ask Question", disabled=True)


    # Output areas

    upload_output = widgets.Output()

    qa_output = widgets.Output()
    system_output = widgets.Output(layout={"border": "1px solid black"})


    # Status indicator

    status = widgets.HTML(value="<b>Status:</b> Ready")


    # Connect callbacks

    upload_button.on_click(handle_upload)

    ask_button.on_click(handle_question)

    check_button.on_click(check_health)


def display_widgets():

    system_tab = widgets.VBox(
        [
            widgets.HTML(value="<h3>System Status</h3>"),
            check_button,
            system_output    
        ]
    )


    upload_tab = widgets.VBox(

        [

            widgets.HTML(value="<h3>Document Upload</h3>"),

            file_upload,

            upload_button,

            upload_output,

        ]
    )


    qa_tab = widgets.VBox(

        [

            widgets.HTML(value="<h3>Question Answering</h3>"),

            question_input,

            ask_button,

            qa_output,

        ]
    )


    tabs = widgets.Tab(children=[system_tab, upload_tab, qa_tab])

    tabs.set_title(0, "System")

    tabs.set_title(1, "Upload")

    tabs.set_title(2, "Q&A")


    display(status)
    display(tabs)



def check_health(b):
    system_output.clear_output()  # Clear the output widget
    try:
        response = requests.get(f"{base_url}/health")
        with system_output:  # Use the Output widget context
            print(f"Health Check ({datetime.now()}):")
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        with system_output:
            print(f"Error: {str(e)}")



def handle_upload(b):
    global collection_id
    upload_output.clear_output()  # Clear previous output

    if not file_upload.value:
        with upload_output:
            print("Please select a file first!")
        return

    try:
        # Extract file info from the uploaded data
        file_info = list(file_upload.value)[0]  # Get the first (and only) file
        file_name = file_info["name"]  # Extract file name
        file_content = file_info["content"]  # Extract file content

        # Prepare the file for the POST request
        files = {"file": (file_name, file_content)}

        status.value = "<b>Status:</b> Uploading document..."
        response = requests.post(f"{base_url}/api/documents", files=files)

        if response.status_code == 200:
            collection_id = response.json().get("collection_id", None)
            question_input.disabled = False
            ask_button.disabled = False
            with upload_output:
                print(f"Document uploaded successfully!")
                print(f"Collection ID: {collection_id}")
            status.value = "<b>Status:</b> Document ready"
        else:
            with upload_output:
                print(f"Error: {response.json()}")
            status.value = "<b>Status:</b> Upload failed"

    except Exception as e:
        with upload_output:
            print(f"Error: {str(e)}")
        status.value = "<b>Status:</b> Error occurred"



def handle_question(b):

    if not question_input.value:

        with qa_output:

            print("Please enter a question!")
            return


    try:

        status.value = "<b>Status:</b> Processing question..."

        response = requests.post(

            f"{base_url}/api/query",

            json={

                "question": question_input.value,

                "collection_id": collection_id,

            },
        )


        with qa_output:

            print("\nQuestion:", question_input.value)

            print("\nAnswer:", response.json()["answer"])

            print("\nSources:")

            for idx, source in enumerate(response.json()["sources"], 1):

                print(f"\n{idx}. {source}")


        status.value = "<b>Status:</b> Ready"


    except Exception as e:

        with qa_output:

            print(f"Error: {str(e)}")

        status.value = "<b>Status:</b> Error occurred"



# Initialize the system

def initialize_client(ngrok_url):

    global base_url

    base_url = ngrok_url

    create_widgets()

    display_widgets()



# Usage example

# Replace with your actual ngrok URL

# initialize_client("https://your-ngrok-url")
```