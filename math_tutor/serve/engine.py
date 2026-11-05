"""服务层模型引擎：加载合并模型或底座+LoRA，提供流式/非流式生成。

与训练无关，只读加载，不修改模型文件。0.5B 模型 FP16 约 1GB 显存，
4GB 笔记本可同时运行网页演示与单份评测。
"""
import threading

from math_tutor.common import ROOT


class ModelEngine:
    def __init__(self, model_dir, adapter_dir=None, max_new_tokens=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path = str((ROOT / model_dir).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        kwargs = dict(torch_dtype=torch.float16, trust_remote_code=False)
        if self.device == 'cuda':
            kwargs['device_map'] = {'': 0}
        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if adapter_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str((ROOT / adapter_dir).resolve()))
        self.model.eval()
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens

    def info(self):
        import torch
        if self.device == 'cuda':
            alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        else:
            alloc = total = 0.0
        return dict(status='ok', device=self.device, model=self.model_dir,
                    adapter=self.adapter_dir, max_new_tokens=self.max_new_tokens,
                    vram_allocated_gb=round(alloc, 2), vram_total_gb=round(total, 2))

    def _build_inputs(self, messages):
        import torch
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_tensors='pt', return_dict=True).to(self.model.device)
        # 与本地 chat 一致：长历史不静默截断，直接报错。
        if inputs['input_ids'].shape[1] > 2048:
            raise ValueError('输入超过2048 token，请缩短问题或清空对话')
        return inputs

    def stream(self, messages, max_new_tokens=None, temperature=0.7, top_p=0.9):
        """逐段产出生成文本；生成在后台线程执行，边生成边返回。"""
        import torch
        from transformers import TextIteratorStreamer
        inputs = self._build_inputs(messages)
        do_sample = temperature > 0
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        gen_kwargs = dict(**inputs, streamer=streamer,
                          max_new_tokens=max_new_tokens or self.max_new_tokens,
                          do_sample=do_sample,
                          temperature=temperature if do_sample else None,
                          top_p=top_p if do_sample else None,
                          pad_token_id=self.tokenizer.eos_token_id)
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for text in streamer:
                yield text
        finally:
            thread.join()

    def generate(self, messages, max_new_tokens=None, temperature=0.7, top_p=0.9):
        text = ''.join(self.stream(messages, max_new_tokens, temperature, top_p))
        return text
