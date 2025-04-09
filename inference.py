# 1) 從你改好的 qwen_light 檔案，匯入 模型的Config & 模型
from transformers.models.qwen_light.configuration_qwenlight import QwenLightConfig
from transformers.models.qwen_light.modeling_qwenlight import QwenLightForCausalLM

# 2) Tokenizer 直接用官方的 AutoTokenizer 即可
from transformers import AutoTokenizer

def main():
    # 這是官方釋出的 Qwen2.5 Instruct 權重在 HF 上的路徑
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    local_dir = "transformers/src/transformers/models/qwen_light"  # 用來緩存從雲端下載的檔案

    # (A) 載入 config：使用你自己的 QwenLightConfig，但從雲端抓 config.json 記得本地要覆蓋
    config = QwenLightConfig.from_pretrained( # from_pretrained 會從官方下載，而跑在QwenLightConfig裡
        model_name,
        cache_dir=local_dir
    )

    # (B) 用你自己的 QwenLightForCausalLM，載入官方預訓練權重
    model = QwenLightForCausalLM.from_pretrained( # from_pretrained 會從官方下載，而跑在QwenLightForCausalLM裡
        model_name,
        config=config,
        cache_dir=local_dir,
        torch_dtype="auto",
        device_map="auto",  # 如果有多GPU / accelerate / etc.
    )

    # (C) Tokenizer 用官方的就好
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=local_dir
    )

    # (D) 做個簡單推理測試
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

if __name__ == "__main__":
    main()
