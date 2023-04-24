import os
import openai
from transformers import pipeline
from flask import Flask, jsonify
import json
from flask_cors import CORS
import os
import PIL
import requests
import replicate
import urllib.request
from dotenv import load_dotenv

# app = Flask(__name__)
# CORS(app)

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
os.environ["REPLICATE_API_TOKEN"]= os.getenv("REPLICATE_KEY")
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
# @app.route('/run_app', methods=['POST'])
def main():

    # get images from users drawing
  
    curr_image_link = "https://i.imgur.com/4p0DDQ6s.jpg"

    # download image
    PILImage = download_image(curr_image_link)
    urllib.request.urlretrieve(curr_image_link, "pic.png")
    curr_image = open("pic.png", "rb")
    # caption the image
    
    generated_text = image_to_text(PILImage)[0]['generated_text']
    print(generated_text)

    # ask GPT for Bob Ross-styled suggestions
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", 
             "content": f"My art piece looks like {generated_text}. Can you give me five suggestions in the style of Bob Ross? Separate each response on a new line."}
        ]
        )

    suggestions = completion.choices[0].message.content.splitlines()
    suggestions = list(filter(None, suggestions))
    # write suggestions to suggestions.json
    with open("suggestions.json", "w") as outfile:
        json.dump(suggestions, outfile)
    # suggestions = "make it cute!".splitlines()

    # retrieve examples of the suggestions applied to images  (image edit)
    image_edits = []

    for s in suggestions:
        if (s == ""):
            continue
        output = replicate.run("arielreplicate/instruct-pix2pix:10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430",
                                input={"input_image": curr_image,
                                       "instruction_text": s})
        image_edits.append(output)
    
    # write suggestions to output.json
    with open("output.json", "w") as outfile:
        json.dump(image_edits, outfile)

    # return jsonify({'reply':'success'})

main()

# if __name__ == "__main__":
#     app.run(debug=True)
