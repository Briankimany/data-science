import xml.etree.ElementTree as ET

# Load the XML file
tree = ET.parse("/home/brian/forex/steve_binary/special/VIBRANIUM DBOT V1.01.xml")
root = tree.getroot()

# Access elements within the XML
for variable in root.findall(".//variables/variable"):
    id = variable.get("id")
    name = variable.text
    print(f"Variable ID: {id}, Name: {name}")

# You can access other elements in a similar way
