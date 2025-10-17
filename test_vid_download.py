import os
import time

import requests


def main() -> None:
    endpoint = os.environ["AZURE_OPENAI_API_BASE"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    videos_url = f"{endpoint}/openai/v1/videos"

    headers = {"api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": "sora-2",
        "prompt": "a cat playing piano in jazz bar",
        "seconds": "8",
        "size": "1280x720",
    }

    response = requests.post(videos_url, headers=headers, json=payload)
    response.raise_for_status()
    creation_payload = response.json()
    print("Full response JSON:", creation_payload)

    video_id = creation_payload.get("id")
    if not video_id:
        raise RuntimeError("No video id returned from creation response.")
    print(f"Job created: {video_id}")

    status_url = f"{videos_url}/{video_id}"
    status = None
    while status not in ("completed", "failed", "cancelled"):
        time.sleep(5)
        status_response = requests.get(status_url, headers=headers)
        status_response.raise_for_status()
        job_state = status_response.json()
        status = job_state.get("status")
        print(f"Job status: {status}")

    if status != "completed":
        raise RuntimeError(f"Job didn't succeed. Status: {status}")

    download_url = f"{status_url}/content"
    download_response = requests.get(
        download_url, headers=headers, params={"variant": "video"}
    )
    download_response.raise_for_status()

    output_filename = "output.mp4"
    with open(output_filename, "wb") as file:
        file.write(download_response.content)
    print(f'Generated video saved as "{output_filename}"')


if __name__ == "__main__":
    main()
