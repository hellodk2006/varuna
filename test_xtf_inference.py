from app.xtf_inference import run_xtf_mission


XTF = r"E:\SIH26057\VARUNA_DEMO_MISSION.xtf"


# IMPORTANT:
# For this first test, explicitly use the currently validated
# mixed v2 model. Do not wait for the final GhostVision model.
MODEL = (
    r"E:\SIH26057\runs\detect\mixed_varuna_v2_50ep"
    r"\weights\best.pt"
)

def main():

    results = run_xtf_mission(
        xtf_path=XTF,
        model_path=MODEL,
        confidence=0.25,
        iou=0.45
    )
    print()
    print("FINAL SUMMARY")
    print("------------------------------")

    print(
        "Mission:",
        results["mission"]
    )

    print(
        "Frames:",
        results["processed_frame_count"]
    )

    print(
        "Detections:",
        results["detection_count"]
    )

    print(
        "Class counts:",
        results["class_counts"]
    )

    print(
        "JSON:",
        results["results_json_path"]
    )


if __name__ == "__main__":
    main()