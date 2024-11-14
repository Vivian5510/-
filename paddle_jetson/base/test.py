# api_key = 'sk-jIkUcN1hi0EyahTxC6988bE772D949A89cC70822F26e3531'
# image_path = 'F:/Material/Pictures/1.png'  # 替换为你本地图片的路径
import requests
import json
import logging
import cv2

def load_image(image_path):
    # 使用 OpenCV 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
    return image

def analyze_image(image):
    # 在这里实现你的图片分析逻辑
    # 示例返回值
    return {
        "hat": False,
        "glasses": True,
        "sleeve": "Long",
        "color_upper": "Plaid",
        "color_lower": "Pattern",
        "clothes_upper": "LongCoat",
        "clothes_lower": "trousers",
        "boots": True,
        "bag": "Backpack",
        "holding": False,
        "age": "Young",
        "sex": "Female",
        "direction": "Front"
    }

def send_request_4(kw):
    api_key = 'sk-jIkUcN1hi0EyahTxC6988bE772D949A89cC70822F26e3531'
    try:
        api_url = 'https://api.apiyi.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': "dall-e-2",
            'messages': [{"role": "system", "content": kw}]
        }
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f'Error: Received status code {response.status_code}'
    except Exception as e:
        logging.info(e)
        return 'An error occurred while sending the request'

def analyze_and_describe_image(image_path):
    image = load_image(image_path)
    features = analyze_image(image)
    
    # 构造描述信息
    description = f'''
    'hat': {features['hat']},					
    'glasses': {features['glasses']},
    'sleeve': '{features['sleeve']}',
    'color_upper': '{features['color_upper']}',
    'color_lower': '{features['color_lower']}',
    'clothes_upper': '{features['clothes_upper']}',
    'clothes_lower': '{features['clothes_lower']}',
    'boots': {features['boots']},
    'bag': '{features['bag']}',
    'holding': {features['holding']},
    'age': '{features['age']}',
    'sex': '{features['sex']}',
    'direction': '{features['direction']}'
    '''
    
    # 调用 GPT-4o API 生成最终描述
    response = send_request_4(description)
    print(response)

if __name__ == '__main__':
    image_path = 'F:/Material/Pictures/3.png'  # 替换为你本地图片的路径
    analyze_and_describe_image(image_path)
