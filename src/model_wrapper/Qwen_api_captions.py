import base64
from io import BytesIO
import traceback
from PIL import Image
import json
import time
import os
from openai import AsyncOpenAI
from dashscope import MultiModalConversation
import asyncio


def encode_image(image_files):

    base_img = []
    for file in image_files:
        encoded_string = base64.b64encode(file).decode("utf-8")
        base_img.append(encoded_string)
    return base_img


def _get_dashscope_api_key():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key is None or api_key.strip() == "":
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    return api_key


def _parse_json_array(raw):
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    captions = json.loads(raw)

    if not isinstance(captions, list):
        raise ValueError("DashScope response is not a JSON array: {}".format(captions))

    return captions


def _call_dashscope_with_retry(model, messages, max_retry=3):
    last_error = None

    for i in range(max_retry):
        try:
            response = MultiModalConversation.call(
                api_key=_get_dashscope_api_key(),
                model=model,
                messages=messages
            )
            return response

        except Exception as e:
            last_error = e
            print("[DashScope] retry {}/{} after error: {}".format(i + 1, max_retry, e))
            time.sleep(2 ** i)

    raise last_error


def generate_caption(image_file, temperature=0.7):
    n = len(image_file)
    system_prompt = f"""
            You are an image understanding assistant. Your task is to generate one concise and detailed description per image that focuses on:

            1. **Key objects and their core attributes**  
            - List every object with simple nouns **and** one or two attributes (e.g., “yellow slide, medium-sized, plastic”; “Springer Spaniel dog, tricolor coat”; “gas station building, metal canopy”).  
            
            2. **Object quantities and groupings**  
            - Specify number if more than one or if it’s a cluster (e.g., “three children”, “a row of parked cars”).  
            
            3. **Precise spatial relationships**  
            - Describe relative positions, distances or directions (e.g., “the dog sits immediately to the right of the slide”, “the building stands in the distant background, slightly left of center”).  
            
            4. **Object states or actions**  
            - Note any visible activity or condition (e.g., “children sliding down”, “pump nozzles hanging idle”, “car doors open”).  
                
            5. **Avoid opinions or irrelevancies**  
            - Use plain factual language. Do not include judgments, emotional tone words, or fine-grained internal part details.  
        
            User will supply {n} images; you must return exactly one well-structured description string for every image(no quotation marks).
            For each image, return exactly only one description item. 
            You are not allowed to return a blank caption string or more than {n} caption for {n} images.
            Return your answer **only** as a JSON array of caption strings (no prose outside the array), like:
            Example output exactly:
            [
                "yellow slide, medium-sized, plastic; slide is in the foreground; children sliding down",
                "......",
                "gas station building, metal canopy; two fuel pumps; pump nozzles hanging idle; canopy in the background"
            ]
            Do not write anything else.
            IMPORTANT INSTRUCTIONS:
            - You must respond **only in valid JSON format**.
            - Do **not** include any extra explanation, commentary, markdown formatting, or newlines.
            - The response must be a **single-line JSON array**, not a stringified object or object.
    """
    
    
    captions = []
    user_content = []
    for img_b64 in image_file:
            user_content.append({'image': 'data:image/png;base64,' + img_b64})
    messages = [{"role": "system",
                "content": [system_prompt]},
                {'role':'user','content': user_content}]
    
    try:
        response = _call_dashscope_with_retry(
            model='qwen-vl-max',
            messages=messages,
            max_retry=3
        )

        # 检查响应是否为空或结构异常
        if not response or "output" not in response:
            raise RuntimeError("DashScope API response is None or missing 'output'")

        # 尝试提取文本内容
        raw = response["output"]["choices"][0]["message"].content[0]["text"].strip()

        # 解析 JSON 格式
        captions = _parse_json_array(raw)

        if len(captions) != len(image_file):
            raise ValueError(
                f"For batch of {len(image_file)} images, expected {len(image_file)} captions but got {len(captions)}:\n{captions}"
            )

        return captions

    except Exception as e:
        print("[generate_caption] Failed to get or parse caption:")
        traceback.print_exc()
        # 返回默认描述（防止程序崩溃）
        return ["Description time out. [ServerError]"] * len(image_file)


def generate_event_caption(image_file, temperature=0.3):
    n = len(image_file)
    system_prompt = f"""
            You are an event-camera image understanding assistant.

            The user will supply {n} event accumulation images captured during UAV movement.
            These images are not ordinary RGB photos.

            Important visual meaning:
            - Red regions indicate positive brightness-change events.
            - Blue regions indicate negative brightness-change events.
            - Brighter regions indicate denser event activity.
            - Dark background indicates little or no event activity.
            - The image mainly shows motion-induced edges, brightness changes, and event-active regions.

            Your task is to generate one concise and factual description per event image.

            Focus on:
            1. **Spatial distribution of event activity**
            - Describe where event activity is strong or weak, such as left, right, center, upper-left, lower-right.

            2. **Event structure**
            - Mention whether the activity forms edge-like, boundary-like, clustered, scattered, or sparse patterns.

            3. **Navigation-related interpretation**
            - Explain that strong event regions may indicate motion boundaries, nearby structure edges, or visual changes during the previous UAV movement.

            4. **Avoid over-claiming semantics**
            - Do not claim exact object categories unless they are visually obvious.
            - Do not say the event-active region is the target object.
            - Treat event activity only as auxiliary motion-boundary evidence.

            User will supply {n} event images; you must return exactly one caption string for every image.
            Return your answer **only** as a JSON array of caption strings.
            Do not write anything else.

            Example output exactly:
            [
                "strong event activity appears on the left and lower-left regions; the activity forms edge-like structures, while the center region is mostly dark; this suggests motion-induced boundaries during the previous UAV movement",
                "scattered red and blue event activity appears in the upper-right and middle-right regions; the left side has weak activity; these regions may indicate recent brightness changes or nearby structural edges"
            ]

            IMPORTANT INSTRUCTIONS:
            - You must respond **only in valid JSON format**.
            - Do **not** include markdown formatting.
            - The response must be a **single-line JSON array**.
    """
    
    captions = []
    user_content = []
    for img_b64 in image_file:
            user_content.append({'image': 'data:image/png;base64,' + img_b64})
    messages = [{"role": "system",
                "content": [system_prompt]},
                {'role':'user','content': user_content}]
    
    try:
        response = _call_dashscope_with_retry(
            model='qwen-vl-max',
            messages=messages,
            max_retry=3
        )

        if not response or "output" not in response:
            raise RuntimeError("DashScope API response is None or missing 'output'")

        raw = response["output"]["choices"][0]["message"].content[0]["text"].strip()

        captions = _parse_json_array(raw)

        if len(captions) != len(image_file):
            raise ValueError(
                f"For batch of {len(image_file)} event images, expected {len(image_file)} captions but got {len(captions)}:\n{captions}"
            )

        return captions

    except Exception as e:
        print("[generate_event_caption] Failed to get or parse event caption:")
        traceback.print_exc()
        return ["No reliable event accumulation image caption is available."] * len(image_file)