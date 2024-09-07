import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Define the labels for the diagram
labels = {'100': 'Predictive Analytics',
          '010': 'Consumer Behavior Theory',
          '001': 'Resource-Based View (RBV)',
          '110': 'Personalization & Customer Engagement',
          '101': 'Leveraging Data for Competitive Advantage\n\n',
          '011': '                                                 Understanding & Utilizing Customer Data',
          '111': 'Optimizing E-commerce Strategies'}

# Create the Venn diagram
plt.figure(figsize=(10, 8))
venn = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=('Predictive Analytics', 'Consumer Behavior Theory', 'Resource-Based View'))
##for idx, label in labels.items():
venn.get_label_by_id("100").set_text('Predictive Analytics')
venn.get_label_by_id("010").set_text('              Consumer Behavior Theory')
venn.get_label_by_id("001").set_text('Resource-Based View (RBV)')
venn.get_label_by_id("110").set_text('Personalization & Customer Engagement')
venn.get_label_by_id("101").set_text('Leveraging Data for\n      Competitive Advantage')
venn.get_label_by_id("011").set_text('                                 Understanding & Utilizing\n                 Customer Data')
venn.get_label_by_id("111").set_text('Optimizing E-commerce Strategies')


venn.get_label_by_id("101").set_y(-0.13)
venn.get_label_by_id("101").set_x(-0.45)
venn.get_label_by_id("011").set_y(-0.13)
venn.get_label_by_id("100").set_x(-0.50)
venn.get_label_by_id("110").set_y(0.40)
venn.get_label_by_id("001").set_y(-0.45)
venn.get_label_by_id("111").set_fontsize(12)
venn.get_label_by_id("111").set_y(0.0)
venn.get_label_by_id("111").set_fontweight('bold')
venn.get_label_by_id("111").set_color('navy')

# Set the title
plt.title('Theoretical Frameworks Connected to Research Focus', fontsize=16)

# Save the diagram as an image file
plt.savefig('Theoretical_Framework_Diagram.png')

# Display the diagram
plt.show()
