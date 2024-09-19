import json

def parse_qa_string(qa_string):
    qa_list = []
    lines = qa_string.strip().split('\n\n')
    for line in lines:
        parts = line.split('Answer:')
        if len(parts) == 2:
            query = parts[0].replace('Query: ', '').strip()
            answer = parts[1].strip()
            qa_list.append({"Query": query, "Answer": answer})
    return qa_list

def main():
    input_file = '/data/SyL/Event_RGB/data_process/thun_02_a_QA_Dataset.json'
    output_file = '/data/SyL/Event_RGB/data_process/thun_02_a_QADataset.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'QA' in item:
            qa_string = item['QA']
            qa_list = parse_qa_string(qa_string)
            item['QA'] = qa_list

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()