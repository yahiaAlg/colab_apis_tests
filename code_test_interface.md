```python
from pprint import pprint
import ipywidgets as widgets
from IPython.display import display, HTML
import requests
import json
from datetime import datetime

# Global variables
base_url = None
file_upload = None
code_input = None
language_input = None
task_dropdown = None
description_input = None
performance_checkbox = None
security_checkbox = None
action_button = None
check_button = None
output_area = None
system_output = None
status = None
collection_info = None


def create_widgets():
    global file_upload, code_input, language_input, task_dropdown, description_input
    global performance_checkbox, security_checkbox, action_button, output_area
    global system_output, status, check_button, collection_info

    # File upload widget
    file_upload = widgets.FileUpload(
        description="Upload Code File",
        accept=".py,.js,.java,.cpp,.c,.rb,.go,.rs,.php,.cs",
        multiple=False,
    )

    # Code input
    code_input = widgets.Textarea(
        description="Code:",
        placeholder="Enter your code here...",
        layout=widgets.Layout(width="90%", height="200px"),
    )

    # Description input for code generation
    description_input = widgets.Textarea(
        description="Description:",
        placeholder="Enter requirements for code generation...",
        layout=widgets.Layout(width="90%", height="100px"),
    )

    # Language input
    language_input = widgets.Dropdown(
        description="Language:",
        options=[
            "python",
            "javascript",
            "java",
            "cpp",
            "c",
            "ruby",
            "go",
            "rust",
            "php",
            "csharp",
        ],
        value="python",
    )

    # Task selection
    task_dropdown = widgets.Dropdown(
        description="Task:", options=["generate", "debug", "document"], value="debug"
    )

    # Analysis options
    performance_checkbox = widgets.Checkbox(
        value=False,
        description="Include Performance Analysis",
        layout=widgets.Layout(margin="10px 0px"),
    )

    security_checkbox = widgets.Checkbox(
        value=False,
        description="Include Security Analysis",
        layout=widgets.Layout(margin="10px 0px"),
    )

    # Buttons
    action_button = widgets.Button(description="Process Code")
    check_button = widgets.Button(description="Check Health")

    # Output areas
    output_area = widgets.Output(
        layout={"border": "1px solid #ddd", "padding": "10px", "margin": "10px 0px"}
    )
    system_output = widgets.Output(
        layout={"border": "1px solid #ddd", "padding": "10px"}
    )

    # Collection info area
    collection_info = widgets.HTML(
        value="<b>Collection ID:</b> None", layout={"margin": "10px 0px"}
    )

    # Status indicator
    status = widgets.HTML(value="<b>Status:</b> Ready")

    # Connect callbacks
    action_button.on_click(handle_action)
    check_button.on_click(check_health)
    task_dropdown.observe(on_task_change, names="value")


def on_task_change(change):
    """Update UI elements based on selected task"""
    if change.new == "generate":
        description_input.layout.visibility = "visible"
        code_input.layout.visibility = "hidden"
        performance_checkbox.layout.visibility = "hidden"
        security_checkbox.layout.visibility = "hidden"
    elif change.new == "debug":
        description_input.layout.visibility = "hidden"
        code_input.layout.visibility = "visible"
        performance_checkbox.layout.visibility = "visible"
        security_checkbox.layout.visibility = "visible"
    else:  # document
        description_input.layout.visibility = "hidden"
        code_input.layout.visibility = "visible"
        performance_checkbox.layout.visibility = "hidden"
        security_checkbox.layout.visibility = "hidden"


def display_widgets():
    system_tab = widgets.VBox(
        [widgets.HTML(value="<h3>System Status</h3>"), check_button, system_output]
    )

    code_input_tab = widgets.VBox(
        [
            widgets.HTML(value="<h3>Code Processing</h3>"),
            task_dropdown,
            description_input,
            code_input,
            widgets.HBox([language_input]),
            widgets.HBox([performance_checkbox, security_checkbox]),
            action_button,
            collection_info,
            output_area,
        ]
    )

    file_upload_tab = widgets.VBox(
        [
            widgets.HTML(value="<h3>File Upload</h3>"),
            file_upload,
            task_dropdown,
            widgets.HBox([performance_checkbox, security_checkbox]),
            action_button,
            collection_info,
            output_area,
        ]
    )

    tabs = widgets.Tab(children=[system_tab, code_input_tab, file_upload_tab])
    tabs.set_title(0, "System")
    tabs.set_title(1, "Code Input")
    tabs.set_title(2, "File Upload")

    display(status)
    display(tabs)


def check_health(b):
    system_output.clear_output()
    try:
        response = requests.get(f"{base_url}/health")
        with system_output:
            print(f"Health Check ({datetime.now()}):")
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        with system_output:
            print(f"Error: {str(e)}")


def handle_action(b):
    output_area.clear_output()
    collection_info.value = "<b>Collection ID:</b> None"

    try:
        status.value = "<b>Status:</b> Processing..."

        if task_dropdown.value == "debug":
            if file_upload.value:
                file_info = list(file_upload.value.values())[0]
                files = {"file": (file_info.name, file_info.content)}
                params = {
                    "include_performance_analysis": performance_checkbox.value,
                    "include_security_analysis": security_checkbox.value,
                }
                response = requests.post(
                    f"{base_url}/api/debug/file", files=files, params=params
                )
            else:
                payload = {
                    "code": code_input.value,
                    "language": language_input.value,
                    "include_performance_analysis": performance_checkbox.value,
                    "include_security_analysis": security_checkbox.value,
                }
                response = requests.post(f"{base_url}/api/debug", json=payload)

            display_debug_results(response.json())

        else:  # generate or document
            if file_upload.value:
                file_info = list(file_upload.value.values())[0]
                files = {"file": (file_info.name, file_info.content)}
                params = {"task": task_dropdown.value}
                response = requests.post(
                    f"{base_url}/api/code/file", files=files, params=params
                )
            else:
                payload = {
                    "code": code_input.value,
                    "task": task_dropdown.value,
                    "language": language_input.value,
                    "description": (
                        description_input.value
                        if task_dropdown.value == "generate"
                        else None
                    ),
                }
                response = requests.post(f"{base_url}/api/code/generate", json=payload)

            result = response.json()
            collection_info.value = (
                f"<b>Collection ID:</b> {result.get('collection_id', 'None')}"
            )

            with output_area:
                print(f"=== {task_dropdown.value.title()} Results ===\n")
                print(result.get("result", "No result available"))

        status.value = "<b>Status:</b> Ready"

    except Exception as e:
        with output_area:
            print(f"Error: {str(e)}")
        status.value = "<b>Status:</b> Error occurred"


def display_debug_results(result):
    with output_area:
        print("=== Analysis Results ===\n")

        if result.get("issues"):
            print("Issues Found:")
            for issue in result["issues"]:
                severity_color = {
                    "error": "red",
                    "warning": "orange",
                    "info": "blue",
                }.get(issue["severity"], "black")

                line_info = (
                    f"Line {issue['line_number']}: " if issue.get("line_number") else ""
                )
                display(
                    HTML(
                        f"<div style='color: {severity_color}'><b>{issue['severity'].upper()}:</b> {line_info}{issue['message']}</div>"
                    )
                )
                print(f"Suggested Fix: {issue['suggested_fix']}\n")

        print("\n=== Fixed Code ===")
        print(result.get("fixed_code", "No fixes required"))

        if result.get("performance_analysis"):
            print("\n=== Performance Analysis ===")
            print(result["performance_analysis"])

        if result.get("security_analysis"):
            print("\n=== Security Analysis ===")
            print(result["security_analysis"])


def initialize_client(ngrok_url):
    global base_url
    base_url = ngrok_url
    create_widgets()
    display_widgets()
    # Hide description input initially
    description_input.layout.visibility = "hidden"


# Usage example
# initialize_client("https://your-ngrok-url")
```