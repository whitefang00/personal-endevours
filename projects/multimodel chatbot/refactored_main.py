
import os
from openai import OpenAI
import gradio as gr
from openai import OpenAI
import google.generativeai
import anthropic

ANTHROPIC_API_KEY='...'
go = "..."
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '...')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY',ANTHROPIC_API_KEY)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', go)


# Imitializing llm apis
openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()
system_message = "Your a helpful assistant"

MODEL_MAP = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4-mini": "gpt-4-mini",
    "anthropic-chat": "Anthropic",
    "gemini-chat": "gemini-chat"
}
system_message = "You are a helpful assistant."

def chat(message, history, model):
    if not message.strip():
        return "Please enter a valid message.", history

    messages = [{"role": "system", "content": system_message}]
    
    for user_message, assistant_message in history:
        if user_message.strip():
            messages.append({"role": "user", "content": user_message})
        if assistant_message.strip():
            messages.append({"role": "assistant", "content": assistant_message})
    
    messages.append({"role": "user", "content": message})

    if model.startswith("gpt-"): 
        response = openai.chat.completions.create(model=model, messages=messages, stream=False)
        assistant_response = response.choices[0].message.content

    elif model == "Anthropic":  
        response = claude.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=200,
            system=system_message,
            messages=messages[1:]
        )
        
        

        if hasattr(response, 'completion'):
            assistant_response = response.completion
        elif 'error' in response:
            assistant_response = f"Anthropic API Error: {response['error']['message']}"
        else:
            assistant_response = response.content[0].text

    elif model == "gemini-chat": 
        gemini = google.generativeai.GenerativeModel(
                model_name='gemini-1.5-flash',
                system_instruction=system_message
            )
        result =gemini.generate_content(message)
        assistant_response =result.text
        
    else:
        assistant_response = "Model not supported."

    return assistant_response, history + [(message, assistant_response)]


def chat_handler(message, dropdown_value, history):
    if dropdown_value is None:
        dropdown_value = "gpt-3.5-turbo"
    model = MODEL_MAP[dropdown_value]
    response, updated_history = chat(message, history, model)
    formatted_history = "\n".join([f"You: {user}\nAssistant: {assistant}" for user, assistant in updated_history])
    return response, formatted_history, updated_history, ""

# Function to clear the chat history
def clear_history():
    return "", "", []

# Function to clear the input box
def clear_input():
    return ""

# Function to handle model change confirmation
def confirm_model_change(dropdown_value, history):
    selected_model = dropdown_value if dropdown_value else "gpt-3.5-turbo"
    message = f"Model changed to: {selected_model}"
    return history + [(message, "")], message

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("### Chat with Different Models and Actions")
    
    chatbox = gr.Textbox(label="Type your message here", placeholder="Enter your message...", lines=2)
    dropdown = gr.Dropdown(choices=list(MODEL_MAP.keys()), label="Select Model", value="gpt-3.5-turbo")  # Default to gpt-3.5-turbo
    
    with gr.Row():
        submit_button_chat = gr.Button("Send Chat", scale=1)
        clear_history_button = gr.Button("Clear History", scale=1)
        clear_input_button = gr.Button("Clear Input", scale=1)
        confirm_model_button = gr.Button("Model Change", scale=1)
  
    output_box = gr.Textbox(label="Chat History", interactive=False, lines=10)

    # State to maintain chat history
    chat_history = gr.State([])

    # Handle chat button click
    submit_button_chat.click(chat_handler, inputs=[chatbox, dropdown, chat_history], outputs=[output_box, output_box, chat_history, chatbox])
    
    # Handle confirm model change button click
    confirm_model_button.click(confirm_model_change, inputs=[dropdown, chat_history], outputs=[chat_history, output_box])

    # Handle clear history button click
    clear_history_button.click(clear_history, outputs=[output_box, chatbox, chat_history])

    # Handle clear input button click
    clear_input_button.click(clear_input, outputs=[chatbox])

# Launch the interface
demo.launch()


