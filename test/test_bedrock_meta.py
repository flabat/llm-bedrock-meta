import llm
from llm.plugins import pm
from llm_bedrock_meta import BedrockLlama


def test_plugin_is_installed():
    plugins = pm.get_plugins()
    assert "llm_bedrock_meta" in {mod.__name__ for mod in plugins}


def test_build_llama2_chat_prompt_conversation():
    model = BedrockLlama("bedrock-llama2-13b")
    conversation = model.conversation()
    conversation.responses = [
        llm.Response.fake(model, "prompt 1", "System Prompt", "response 1"),
        llm.Response.fake(model, "prompt 2", None, "response 2"),
        llm.Response.fake(model, "prompt 3", None, "response 3"),
    ]
    bits = model.build_llama2_chat_prompt(llm.Prompt("prompt 4", model), conversation)
    assert bits == [
        "<s>[INST] ",
        "<<SYS>>\nSystem Prompt\n<</SYS>>\n\n",
        "prompt 1 [/INST] ",
        "response 1 </s>",
        "<s>[INST] ",
        "prompt 2 [/INST] ",
        "response 2 </s>",
        "<s>[INST] ",
        "prompt 3 [/INST] ",
        "response 3 </s>",
        "<s>[INST] ",
        "prompt 4 [/INST] ",
    ]