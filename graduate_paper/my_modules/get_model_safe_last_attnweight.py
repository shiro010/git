import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from DynamicScalingLlamaAttention_last_attnweight import DynamicScalingLlamaAttention

CACHED_MODEL = None
CACHED_TOKENIZER = None

def get_model_safe(model_path):
    global CACHED_MODEL, CACHED_TOKENIZER

    if CACHED_MODEL is not None:
        print("✅ Model already loaded. Using cached model.")
        return CACHED_MODEL, CACHED_TOKENIZER

    print("⏳ Loading new model... (This may take a while)")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="eager",
        dtype=torch.float16,
        use_cache=True,
        trust_remote_code=True
    )

    print("🔄 Swapping Attention Modules to DynamicScalingLlamaAttention...")

    target_layers = list(range(14,20))

    # モデル設定
    model.config.hidden_scale_config = {
        "target_layers": target_layers,
        "target_dims": [2393],
        "factor": 1,
        "last_recompute_tokens": 1,
        "change_value": False
    }

    # 全レイヤー入れ替え
    for i, layer in enumerate(model.model.layers):
        if i in target_layers:
            original_attn = layer.self_attn
            target_device = original_attn.q_proj.weight.device
            target_dtype = original_attn.q_proj.weight.dtype

            new_attn = DynamicScalingLlamaAttention(config=model.config, layer_idx=i)

            # originalの情報をコピー
            new_attn.load_state_dict(original_attn.state_dict(), strict=False)
            new_attn.to(device=target_device, dtype=target_dtype)
            layer.self_attn = new_attn
        else: pass

    CACHED_MODEL = model
    CACHED_TOKENIZER = tokenizer

    print("✅ Model loaded and patched successfully.")
    return CACHED_MODEL, CACHED_TOKENIZER

def test_patched_model_sanity(model_path="meta-llama/Llama-2-7b-chat-hf"):
    print("🔬 Starting Model Sanity Check...")
    print("-" * 40)

    # 1. モデル取得 (get_model_safeを利用)
    try:
        model, tokenizer = get_model_safe(model_path)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"❌ Failed to load model:\n{error_msg}")
        return

    # 2. レイヤーの入れ替え確認
    target_layer = model.model.layers[15] # 15層目のattentionが正しく割り当てられているかを確認
    attn_class_name = target_layer.self_attn.__class__.__name__

    print(f"🧐 Checking Layer 0 Attention Class...")
    print(f"   -> Detected: {attn_class_name}")

    if "DynamicScaling" in attn_class_name:
        print("   ✅ SUCCESS: Attention module is correctly swapped!")
    else:
        print("   ⚠️ WARNING: Still using original Attention. Patching failed.")

    # 3. 単純な生成テスト (スケーリングなし)
    print("\n🏃 Running Basic Generation Test (No Scaling)...")
    input_text = "Hello, my name is"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    # 念のため scale_map を None に設定
    if hasattr(model.config, "current_scale_map"):
        model.config.current_scale_map = None

    try:
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        print(f"   -> Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print("   ✅ Basic generation passed.")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"❌ Basic generation failed:\n{error_msg}")
        return

    # 4. スケーリング注入テスト
    print("\n💉 Running Scaled Generation Test (With Dummy Map)...")

    # ダミーのスケールマップ作成 (全トークンを 0.5倍 にするマップ)
    seq_len = input_ids.shape[1]
    dummy_scale_map = torch.full((1, seq_len), 0, device=model.device, dtype=torch.float16)

    # 設定注入
    model.config.current_scale_map = dummy_scale_map

    try:
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        print(f"   -> Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print("   ✅ Scaled generation passed! (Logic seems correct)")
    except Exception as e:
        print(f"   ❌ Scaled generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 40)
    print("🎉 Test Complete.")

# 要らないcacheをクリアしてVRAMを確保する
def remove_cache():
  global CACHED_MODEL, CACHED_TOKENIZER
  CACHED_MODEL = None
  CACHED_TOKENIZER = None

  import gc
  import torch
  gc.collect()
  torch.cuda.empty_cache()

  print("🧹 Cache CLEARED. 次の実行でモデルが『最初から』読み込まれます。")

