from __future__ import annotations

from pathlib import Path
from typing import List


def format_patch(patch: List[str], patch_index: int) -> str:
    """Format a single patch with header and data rows."""
    header = f"===== Patch {patch_index} =====\n"
    data = "\n".join(patch)
    return header + data


def create_forecast_prompt_kr(
    all_patches: List[List[str]],
    num_input_patches: int,
    num_predictions: int,
    start_index: int | None = None,
    patch_structure_file: Path | None = None,
) -> str:
    """Create a Korean forecasting prompt asking to predict N patches from M input patches.
    
    Args:
        all_patches: List of all available patches, each patch is a list of strings (date,open,high,low,close)
        num_input_patches: Number of past patches to use as input (M)
        num_predictions: Number of patches to predict (N)
        start_index: Optional starting index. If None, uses the last M patches from all_patches
        patch_structure_file: Optional path to patch structure description file
    
    Returns:
        Formatted Korean prompt string
    """
    if num_input_patches <= 0:
        raise ValueError("num_input_patches must be greater than 0")
    if num_predictions <= 0:
        raise ValueError("num_predictions must be greater than 0")
    
    # Determine input patches
    if start_index is None:
        # Use last M patches
        if len(all_patches) < num_input_patches:
            raise ValueError(f"Not enough patches. Need at least {num_input_patches}, got {len(all_patches)}")
        input_patches = all_patches[-num_input_patches:]
        actual_start_index = len(all_patches) - num_input_patches
    else:
        # Use patches starting from start_index
        if start_index + num_input_patches > len(all_patches):
            raise ValueError(
                f"Not enough patches from index {start_index}. "
                f"Need {num_input_patches} patches, but only {len(all_patches) - start_index} available."
            )
        input_patches = all_patches[start_index:start_index + num_input_patches]
        actual_start_index = start_index
    
    structure_path = patch_structure_file or Path(__file__).parent.parent / "prompts" / "patch_structure_kr.txt"
    
    instruction_template = """다음은 암호화폐 시세 데이터를 patch 단위로 나눈 것입니다. 아래에 제시된 {m}개의 patch를 참고하여, 다음 {n}개의 patch에 해당하는 시세 정보를 예측하여 같은 형식으로 작성해주세요.

각 patch는 날짜 순서대로 연속적으로 구성되어 있으며, 예측할 patch들은 마지막 입력 patch의 다음 날짜부터 시작해야 합니다.

입력 patch:
{input_patches}

**중요**: 위 입력 patch들을 바탕으로 다음 {n}개의 patch를 예측해주세요. 

**응답 형식 규칙**:
1. 설명이나 추가 문장 없이 오로지 예측된 patch 데이터만 출력하세요.
2. 각 patch는 다음 형식을 정확히 따라야 합니다:
   - Patch 헤더: "===== Patch {{인덱스}} =====" (인덱스는 입력 patch의 마지막 인덱스 다음부터 시작)
   - 데이터 행: 각 행은 "날짜,시가,고가,저가,종가" 형식
   - Patch 간 구분: 빈 줄로 구분

3. 응답 예시 형식 (다음처럼 정확히 출력):
===== Patch {{시작_인덱스}} =====
{{날짜1}},{{시가1}},{{고가1}},{{저가1}},{{종가1}}
{{날짜2}},{{시가2}},{{고가2}},{{저가2}},{{종가2}}
...

===== Patch {{다음_인덱스}} =====
...

**주의**: 응답에는 예측된 patch 데이터 외의 어떤 설명, 주석, 또는 추가 텍스트도 포함하지 마세요. 오로지 위 형식의 patch 데이터만 출력하세요."""

    # Load patch structure description if file exists
    structure_desc = ""
    if structure_path and structure_path.exists():
        structure_desc = structure_path.read_text(encoding="utf-8")
        structure_desc = "\n\n## Patch 구조 설명\n\n" + structure_desc

    # Format input patches with correct indices
    formatted_patches_list = []
    for i, patch in enumerate(input_patches):
        formatted_patches_list.append(format_patch(patch, actual_start_index + i))
    
    input_patches_str = "\n\n".join(formatted_patches_list)

    # Create the full prompt
    prompt = instruction_template.format(
        m=len(input_patches),
        n=num_predictions,
        input_patches=input_patches_str,
    )
    
    if structure_desc:
        prompt = structure_desc + "\n\n" + prompt

    return prompt


def create_forecast_prompt_en(
    all_patches: List[List[str]],
    num_input_patches: int,
    num_predictions: int,
    start_index: int | None = None,
    patch_structure_file: Path | None = None,
) -> str:
    """Create an English forecasting prompt asking to predict N patches from M input patches.
    
    Args:
        all_patches: List of all available patches, each patch is a list of strings (date,open,high,low,close)
        num_input_patches: Number of past patches to use as input (M)
        num_predictions: Number of patches to predict (N)
        start_index: Optional starting index. If None, uses the last M patches from all_patches
        patch_structure_file: Optional path to patch structure description file
    
    Returns:
        Formatted English prompt string
    """
    if num_input_patches <= 0:
        raise ValueError("num_input_patches must be greater than 0")
    if num_predictions <= 0:
        raise ValueError("num_predictions must be greater than 0")
    
    # Determine input patches
    if start_index is None:
        # Use last M patches
        if len(all_patches) < num_input_patches:
            raise ValueError(f"Not enough patches. Need at least {num_input_patches}, got {len(all_patches)}")
        input_patches = all_patches[-num_input_patches:]
        actual_start_index = len(all_patches) - num_input_patches
    else:
        # Use patches starting from start_index
        if start_index + num_input_patches > len(all_patches):
            raise ValueError(
                f"Not enough patches from index {start_index}. "
                f"Need {num_input_patches} patches, but only {len(all_patches) - start_index} available."
            )
        input_patches = all_patches[start_index:start_index + num_input_patches]
        actual_start_index = start_index
    
    structure_path = patch_structure_file or Path(__file__).parent.parent / "prompts" / "patch_structure_en.txt"
    
    instruction_template = """The following is cryptocurrency price data divided into patches. Based on the {m} patches provided below, please predict the price information for the next {n} patches and write them in the same format.

Each patch is structured consecutively in date order, and the predicted patches should start from the day after the last date in the input patches.

Input patches:
{input_patches}

**IMPORTANT**: Based on the input patches above, please predict the next {n} patches.

**Response Format Rules**:
1. Output ONLY the predicted patch data without any explanations, comments, or additional text.
2. Each patch must follow this exact format:
   - Patch header: "===== Patch {{index}} =====" (indices start from the index after the last input patch index)
   - Data rows: Each row in format "date,open,high,low,close"
   - Patch separator: Blank line between patches

3. Example response format (output exactly like this):
===== Patch {{start_index}} =====
{{date1}},{{open1}},{{high1}},{{low1}},{{close1}}
{{date2}},{{open2}},{{high2}},{{low2}},{{close2}}
...

===== Patch {{next_index}} =====
...

**WARNING**: Do not include any explanations, comments, or additional text in your response. Output ONLY the patch data in the format above."""

    # Load patch structure description if file exists
    structure_desc = ""
    if structure_path and structure_path.exists():
        structure_desc = structure_path.read_text(encoding="utf-8")
        structure_desc = "\n\n## Patch Structure Description\n\n" + structure_desc

    # Format input patches with correct indices
    formatted_patches_list = []
    for i, patch in enumerate(input_patches):
        formatted_patches_list.append(format_patch(patch, actual_start_index + i))
    
    input_patches_str = "\n\n".join(formatted_patches_list)

    # Create the full prompt
    prompt = instruction_template.format(
        m=len(input_patches),
        n=num_predictions,
        input_patches=input_patches_str,
    )
    
    if structure_desc:
        prompt = structure_desc + "\n\n" + prompt

    return prompt


def load_patches_from_txt(patch_file: Path | str) -> List[List[str]]:
    """Load patches from a text file generated by write_patches_to_txt.
    
    Args:
        patch_file: Path to patch text file (e.g., patches/train/ADAUSDT_patches.txt)
    
    Returns:
        List of patches, each patch is a list of strings (date,open,high,low,close)
    """
    patch_file = Path(patch_file)
    if not patch_file.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_file}")
    
    content = patch_file.read_text(encoding="utf-8")
    patches: List[List[str]] = []
    current_patch: List[str] = []
    
    for line in content.splitlines():
        line = line.strip()
        if not line:
            # Empty line marks end of patch
            if current_patch:
                patches.append(current_patch)
                current_patch = []
            continue
        
        if line.startswith("===== Patch") and line.endswith("====="):
            # Patch header, start new patch if previous one exists
            if current_patch:
                patches.append(current_patch)
                current_patch = []
            continue
        
        # Data row (date,open,high,low,close)
        current_patch.append(line)
    
    # Add last patch if exists
    if current_patch:
        patches.append(current_patch)
    
    return patches


def save_forecast_prompt(
    prompt: str,
    output_path: Path | str,
    encoding: str = "utf-8",
) -> None:
    """Save the forecast prompt to a file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding=encoding)


if __name__ == "__main__":
    # Example usage - load patches from pre-generated text file
    example_patch_file = Path(__file__).parent.parent / "patches" / "train" / "ADAUSDT_patches.txt"
    
    if not example_patch_file.exists():
        print(f"Patch file not found: {example_patch_file}")
        print("Please run extract_patches.sh first to generate patch files.")
        exit(1)
    
    # Load patches from text file
    example_patches = load_patches_from_txt(example_patch_file)
    print(f"Loaded {len(example_patches)} patches from {example_patch_file}")
    
    # Use last 3 patches as input, predict next 2 patches
    prompt_kr = create_forecast_prompt_kr(
        all_patches=example_patches,
        num_input_patches=3,
        num_predictions=2
    )
    prompt_en = create_forecast_prompt_en(
        all_patches=example_patches,
        num_input_patches=3,
        num_predictions=2
    )
    
    print("\n=== Korean Prompt ===")
    print(prompt_kr)
    print("\n\n=== English Prompt ===")
    print(prompt_en)

