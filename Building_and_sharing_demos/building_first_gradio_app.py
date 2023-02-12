
import gradio as gr

def greet(name):
    return "Hello " + name 

# We instantiate the Textbox class
textbox = gr.Textbox(label="Type your name here:", placeholder="Name", lines =2)

demo = gr.Interface(fn=greet, inputs= textbox, outputs="text")

demo.launch(share=True)

