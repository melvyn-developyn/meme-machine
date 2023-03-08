import os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

NUMBER_OF_IMAGES_PER_PROMPT = 5

repo_id = "stabilityai/stable-diffusion-2"
pipe = DiffusionPipeline.from_pretrained(repo_id)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def create_image(prompt):
    image = pipe(prompt, guidance_scale=9, num_inference_steps=25).images[0]
    return image

def main():
    items = open('./prompts.txt', 'r').readlines()

    for item in items:
        try:
            os.mkdir(f"./generated-images")
        except:
            print("An error occurred making the folder, probably because it already exists.")

        for i in range(NUMBER_OF_IMAGES_PER_PROMPT):
            image = create_image(item)
            image_name = f"./generated-images/{item}-{i}.png"
            image.save(image_name)
            print(f"Successfully saved: {image_name}")

    
    print("Finished")

main()
