
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Page configuration
st.set_page_config(
    page_title="LearnWise AI",
    page_icon="📚",
    layout="centered"
)

# Load the LLM model once
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model

# Main heading
st.title("📚 LearnWise AI")
st.subheader("Your Personal AI Learning Assistant")

st.write(
    "Learn concepts, ask questions, and understand "
    "difficult topics with the help of AI."
)

st.divider()

# Select learning mode
mode = st.selectbox(
    "Choose Learning Mode",
    [
        "Explain a Concept",
        "Answer a Question",
        "Simple Summary"
    ]
)

# User input
if mode == "Simple Summary":
    user_input = st.text_area(
        "Paste the text you want to summarize",
        height=180,
        placeholder="Paste your study notes here..."
    )
else:
    user_input = st.text_area(
        "Enter your topic or question",
        height=120,
        placeholder="Example: Explain Machine Learning"
    )

# Generate answer
if st.button("Generate Answer", type="primary"):

    if not user_input.strip():
        st.warning("Please enter a topic, question, or text.")

    else:
        try:
            with st.spinner(
                "Loading AI model and generating your answer..."
            ):
                tokenizer, model = load_model()

                if mode == "Explain a Concept":
                    instruction = (
                        "Explain the following concept in simple "
                        "English for a beginner student. "
                        "Give a definition, key points, and one example."
                    )

                elif mode == "Answer a Question":
                    instruction = (
                        "Answer the student's question clearly "
                        "in simple English. Explain the important "
                        "points and give an example when useful."
                    )

                else:
                    instruction = (
                        "Summarize the following text in simple "
                        "English. Keep the main ideas and important "
                        "points. Do not add unsupported information."
                    )

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are LearnWise AI, a helpful student "
                            "learning assistant. Use simple English, "
                            "clear explanations, and examples. "
                            "If you are unsure, say so."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            instruction + "\n\n" + user_input
                        )
                    }
                ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

                input_length = inputs["input_ids"].shape[1]

                answer_tokens = outputs[0][input_length:]

                answer = tokenizer.decode(
                    answer_tokens,
                    skip_special_tokens=True
                )

            st.success("Answer generated successfully!")

            st.markdown("### 💡 AI Answer")
            st.write(answer)

            st.download_button(
                label="Download Answer",
                data=answer,
                file_name="learnwise_answer.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(
                "Something went wrong while loading the AI model "
                "or generating the answer."
            )

            st.write("Error details:", str(e))
            st.info(
                "Check your internet connection and installed "
                "Python packages, then try again."
            )

st.divider()

st.caption("LearnWise AI | Powered by Qwen2.5 and Transformers")