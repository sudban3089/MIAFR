def calculate_accuracy(original, result, attribute_names):
    def file_to_dict(file_path):
        content_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                file_number = parts[0]  
                values = [int(value) for value in parts[1:]]  
                content_dict[file_number] = values
        return content_dict

    original_dict = file_to_dict(original)
    result_dict = file_to_dict(result)
    
    correct_predictions = {}
    total_predictions = {}

    matching_files = sorted(set(original_dict.keys()) & set(result_dict.keys()))

    female_index = attribute_names.index("Female")

    for file_number in matching_files:
        original_values = original_dict[file_number]
        result_values = result_dict[file_number]
        
        for idx, (original, result) in enumerate(zip(original_values, result_values)):
            if idx == female_index:
                continue

            if result == 0:  
                continue
            
            if idx not in total_predictions:
                total_predictions[idx] = 0
                correct_predictions[idx] = 0
            
            total_predictions[idx] += 1
            if original == result: 
                correct_predictions[idx] += 1

    accuracy = {attribute_names[idx]: (correct_predictions[idx] / total_predictions[idx]) * 100 
                              for idx in total_predictions}
    
    total_correct = sum(correct_predictions.values())
    total_total = sum(total_predictions.values())
    total_accuracy = (total_correct / total_total) * 100
    
    accuracy['Total Accuracy for Blip'] = total_accuracy
    
    return accuracy

attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

original = 'Attribute Results/original.txt'
result = 'Attribute Results/result_blip.txt'

accuracy = calculate_accuracy(original, result, attributes)

for attribute, acc in accuracy.items():
    print(f"{attribute}: {round(acc, 2)}%")
