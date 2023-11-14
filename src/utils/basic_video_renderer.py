import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.animation import FuncAnimation

def render_animation(frames, output_path, fps=5, text=None):
  # frames += 0.5
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  frames = frames * std + mean
  # frames = (frames * 255).astype('uint8')
  fig = plt.figure()
  def update_img(n):
      plt.imshow(frames[n])
  
  ani = FuncAnimation(fig, update_img, frames=len(frames))  
  ani.save(output_path, writer='ffmpeg', fps=fps)
  print('Video saved at', output_path)
