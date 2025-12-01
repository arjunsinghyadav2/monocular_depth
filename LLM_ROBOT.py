import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from call_function import call_function, available_functions
from Robot_Tools.Robot_Motion_Tools import device_close

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    MODEL_ID = "gemini-2.5-flash"
    verbose = False
    max_iter = 20

    # Optional initial prompt from CLI
    initial_prompt = None
    if len(sys.argv) >= 2:
        initial_prompt = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == "--verbose":
        verbose = True

    system_prompt = """
    You are an advanced AI robot control agent with vision and depth perception capabilities.
    You are controlling a Dobot robotic arm with the following capabilities:

    - Complete scene understanding with monocular depth estimation
    - Object detection with size classification (small/large blocks)
    - Color detection (blue, green, yellow, red)
    - Precise pick-and-place operations with rotation support
    - Kinematic awareness through depth perception

    ## INITIALIZATION (First time only)
    When you start for the first time:
    1) Initialize the scene understanding system: initialize_scene_understanding_system()
    2) Connect to the robot: get_dobot_device()
    3) Move to home position: move_to_home()

    ## STANDARD WORKFLOW FOR PICK AND PLACE TASKS

    For ANY pick-and-place command, follow these steps:

    1) SCENE CAPTURE:
       - Use capture_and_analyze_complete_scene() to get a complete scene with:
         * Block detection with size classification (small_blue1, large_red2, etc.)
         * Depth information for kinematic awareness
         * Creates: scene_complete.json and scene_depth.png

       - Display detected blocks to user in format:
         "Detected blocks:
          - small_blue1 at (x, y) depth: 0.XX
          - large_yellow1 at (x, y) depth: 0.XX
          ..."

    2) COMMAND PARSING:
       - Parse natural language commands to identify:
         * Source block (what to pick): size + color (e.g., "small blue block")
         * Target block/location (where to place): size + color or direction
         * Action type: "on top of", "beside", "to the right of", etc.
         * Rotation: if mentioned (e.g., "rotate by 90 degrees")

       - Use parse_block_description() to map descriptions to actual labels

    3) EXECUTION:
       - Use pick_and_place_with_rotation() with parameters:
         * detection_json_path: "captures/scene_complete.json"
         * source_label: detected label (e.g., "small_blue1")
         * target_label: detected label (e.g., "large_red1")
         * placement_type: "on_top" or "beside"
         * direction: "right", "left", "front", "back" (for beside placement)
         * rotation_degrees: 0, 90, 180, 270, etc.

    4) CONFIRMATION:
       - Report completion with details of what was done

    ## EXAMPLE COMMAND MAPPINGS

    1. "Pick up the small blue block and place it in the box on the right"
       → source: small_blue, target: box/reference, placement: beside, direction: right

    2. "Pick up the large yellow block and place it in the box on the left"
       → source: large_yellow, target: box/reference, placement: beside, direction: left

    3. "Pick up a small block and place it on top of the large block"
       → source: any small block, target: large block, placement: on_top

    4. "Pick up the small blue block, rotate by 90 degrees in z, and place it on large red block"
       → source: small_blue, target: large_red, placement: on_top, rotation: 90

    5. "Pick a small yellow block and place it to the right of the red block"
       → source: small_yellow, target: red block, placement: beside, direction: right

    ## DEPTH AWARENESS

    The system uses monocular depth estimation to understand 3D positions:
    - depth_normalized values range from 0 (far) to 1 (close)
    - Use this to verify the robot can reach objects
    - The system automatically adjusts for different heights

    ## IMPORTANT NOTES

    - Always use the enhanced detection system (capture_and_analyze_complete_scene)
    - Size matters: distinguish between "small" and "large" blocks
    - When user says "a block" or "the block", ask for clarification if multiple matches exist
    - For "beside" placement, map natural language directions:
      * "right" → direction: "right"
      * "left" → direction: "left"
      * "in front" → direction: "front"
      * "behind" → direction: "back"
    - Rotation angles: 90° = quarter turn, 180° = half turn, 270° = three-quarter turn

    ## ERROR HANDLING

    - If a block is not found, list available blocks
    - If command is ambiguous, ask for clarification
    - If depth indicates unreachable position, warn the user

    You are proactive, intelligent, and safety-conscious. Always confirm you understand
    the command before executing, especially for complex multi-step operations.
    """

    print("\n================ SYSTEM PROMPT ================\n")
    print(system_prompt.strip(), "\n")

    # Conversation history (user + assistant + tool messages)
    messages = []

    # If we got an initial CLI prompt, use it as the first user message
    if initial_prompt:
        print("\n================ USER PROMPT (CLI) ================\n")
        print(initial_prompt)
        messages.append(
            types.Content(role="user", parts=[types.Part(text=initial_prompt)])
        )
    else:
        # Otherwise, ask interactively for the first input
        user_text = input("\nYou (type 'quit' to exit): ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            return
        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=system_prompt
    )

    func_count = 0

    # ================= INTERACTIVE CONVERSATION LOOP =================
    while True:
        # For each user message, allow multiple tool/model turns
        for i in range(max_iter):
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=messages,
                config=config
            )

            # ------------ MODEL TEXT ------------
            if response.text:
                print("\n================ MODEL TEXT RESPONSE ================\n")
                print(response.text)

            if verbose and response.usage_metadata:
                print(f'prompt = {messages[-1].parts[0].text if messages else ""}')
                print(f'Response = {response.text}')
                print(f'Prompt Token = {response.usage_metadata.prompt_token_count}')
                print(f'Response Token = {response.usage_metadata.candidates_token_count}')

            # Add assistant content to history
            if response.candidates:
                for candidate in response.candidates:
                    if candidate and candidate.content:
                        messages.append(candidate.content)

            # ------------ TOOL CALLS ------------
            if response.function_calls:
                for function_call_part in response.function_calls:
                    func_count += 1
                    fname = getattr(function_call_part, "name", None)
                    fargs = getattr(function_call_part, "args", {})

                    print(f"\n================ FUNCTION CALL #{func_count} ================\n")
                    print(f"Function name: {fname}")
                    print("Arguments (tool prompt):")
                    try:
                        print(json.dumps(fargs, indent=2))
                    except TypeError:
                        print(fargs)

                    # Run tool
                    result = call_function(function_call_part, verbose=True)

                    print(f"\n================ FUNCTION RESULT #{func_count} ================\n")
                    print(result)

                    # Append tool result so the model can see it next iteration
                    messages.append(result)

                # continue inner for-loop to let the model react to the tool results
                continue

            # ---------- NO FUNCTION CALLS -> END OF THIS TURN ----------
            break  # break out of the max_iter loop; ready for next user input

        # ================= ASK FOR NEXT USER INPUT =================
        print("\n================ AWAITING USER INPUT (type 'quit' to exit) ================\n")
        user_text = input("You: ").strip()

        if user_text.lower() in {"quit", "exit", "q"}:
            print("Closing robot connection before exit...")
            try:
                result = device_close()
                print(result)
            except Exception as e:
                print(f"Error closing device: {e}")

            print("Exiting interactive session.")
            break
        print("\n================ USER PROMPT ================\n")
        print(user_text)

        # Add new user message and loop again
        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

if __name__ == "__main__":
    main()

