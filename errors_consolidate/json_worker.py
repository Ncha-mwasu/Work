
def txt_to_json(path):
    text_file = open(path, 'r')
    text_lines = list(text_file.readlines())
    text_file.close()
    text_lines.insert(0, "{\n")
    text_lines.append("}\n")

    json_file = open('errors_parallel.json', 'w')
    json_file.writelines(text_lines)


if __name__ == '__main__':
    txt_to_json('C:/Users/ENGR. B.K. NUHU/Desktop/IMPLEMENTATIONS/sdwsn-new-arima/errors_consolidate_parallel.txt')