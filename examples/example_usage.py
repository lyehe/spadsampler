from pathlib import Path
from src.main import main

if __name__ == "__main__":
    input_path = Path("path/to/your/input/file.tiff")
    output_prefix = Path("output/prefix")
    
    main(
        input_path=input_path,
        output_prefix=output_prefix,
        frame_size=(512, 512),
        start_range=-7,
        end_range=-2,
        process_by_frame=False
    )