#!/usr/bin/env python3
"""
Hydra Config 로딩 테스트 스크립트
실제로 4-bit quantization이 제대로 오버라이드되는지 확인
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def test_config(cfg: DictConfig):
    print("=" * 80)
    print("Hydra Config 로딩 테스트")
    print("=" * 80)

    # Hydra 버전 확인
    import hydra as hydra_module
    print(f"\n[Hydra Version]: {hydra_module.__version__}")

    # 전체 config 출력
    print("\n[전체 Config - DictConfig]")
    print(OmegaConf.to_yaml(cfg))

    # train.py처럼 dict로 변환
    config = OmegaConf.to_container(cfg, resolve=True)

    print("\n[전체 Config - to_container 변환 후]")
    import json
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # Quantization 설정 확인 (DictConfig 직접)
    print("\n" + "=" * 80)
    print("[Quantization 설정 확인 - DictConfig]")
    print("=" * 80)

    quant_config_direct = cfg.get('model', {}).get('quantization', {})
    print(f"DictConfig - load_in_4bit: {quant_config_direct.get('load_in_4bit', False)}")

    # Quantization 설정 확인 (dict 변환 후)
    print("\n" + "=" * 80)
    print("[Quantization 설정 확인 - to_container 후]")
    print("=" * 80)

    quant_config = config.get('model', {}).get('quantization', {})

    load_in_4bit = quant_config.get('load_in_4bit', False)
    load_in_8bit = quant_config.get('load_in_8bit', False)

    print(f"load_in_4bit: {load_in_4bit} (type: {type(load_in_4bit)})")
    print(f"load_in_8bit: {load_in_8bit} (type: {type(load_in_8bit)})")
    print(f"bnb_4bit_quant_type: {quant_config.get('bnb_4bit_quant_type', 'N/A')}")
    print(f"bnb_4bit_use_double_quant: {quant_config.get('bnb_4bit_use_double_quant', 'N/A')}")

    # 결과 판정
    print("\n" + "=" * 80)
    print("[결과]")
    print("=" * 80)

    if load_in_4bit:
        print("✅ SUCCESS: 4-bit quantization이 활성화되었습니다!")
    elif load_in_8bit:
        print("⚠️  INFO: 8-bit quantization이 활성화되었습니다.")
    else:
        print("❌ FAIL: Quantization이 비활성화되었습니다 (bf16 full precision).")

    print("\n사용법:")
    print("  기본:         python test_config.py")
    print("  실험 설정:    python test_config.py +experiment=4x16gb")
    print("  디버그:       python test_config.py +experiment=debug")
    print("  CLI 오버라이드: python test_config.py model.quantization.load_in_4bit=false")
    print()


if __name__ == "__main__":
    test_config()
