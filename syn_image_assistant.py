import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

class CulturalIdentityDiagram:
    def __init__(self, center_circle=" ", categories=None):
        self.center_circle = center_circle
        if categories is None:
            self.categories = ['name', 'address', 'love life', 'phone number']
        else:
            self.categories = categories

    def draw_diagram(self):
        # Create a figure and canvas
        fig, ax = plt.subplots(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # Draw the center ellipse with "Cultural Identity" text
        center_ellipse = Ellipse((0, 0), width=1.0, height=0.5, color='skyblue', ec='black', lw=2)
        ax.add_artist(center_ellipse)
        ax.text(0, 0, self.center_circle, ha='center', va='center', fontsize=14)

        # Draw the outer ellipses based on the categories
        for i, category in enumerate(self.categories):
            angle = i * (360 / len(self.categories)) * (np.pi / 180)  # converting degrees to radians
            x = 1.5 * np.cos(angle)
            y = 1.5 * np.sin(angle)
            
            outer_ellipse = Ellipse((x, y), width=1.0, height=0.5, color='lightgreen', ec='black', lw=2)
            ax.add_artist(outer_ellipse)
            ax.text(x, y, category, ha='center', va='center', fontsize=12)

        # Remove the axes for clarity
        ax.axis('off')

        # Save the figure to a buffer
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)

        # Convert to PIL Image
        image = Image.open(buf)
        plt.close(fig)  # Close the figure to release memory

        return image

# Example usage:
if __name__ == "__main__":
    categories = ['Hobbies', 'Occupation', 'Nationality', 'Family']
    diagram = CulturalIdentityDiagram(categories=categories)
    image = diagram.draw_diagram()

    # You can now use the 'image' object (PIL.Image), e.g., display it or save it
    image.show()  # To display the image
    # image.save('diagram.png')  # To save it as a PNG file
