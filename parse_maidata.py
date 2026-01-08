import re

def parse_note_time(file_path:str,level:int)->list:
    pattern = f'&inote_{level}=(.*?)E'

    with open(file_path, "r",encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(pattern, text,re.DOTALL)
    # print(matches[0])
    track = matches[0]

    pattern = r'\((.*?)\)\{(.*?)\}'
    matches = re.search(pattern, track)
    BPM=float(matches.group(1))
    base_beat=int(matches.group(2))
    base_note_length=60.0/BPM*(base_beat/4.0)
    # print("BPM:",BPM,"base_beat:",base_beat,"base_note_length:",base_note_length)

    pattern = r'.*\{(.*?)\}(.*)'
    current_time = 0

    note_time:list=[]

    for count, line in enumerate(track.splitlines(), start=1):
        # print(f"第{count},{line}")
        matches = re.match(pattern, line)
        beat = int(matches.group(1))
        note = matches.group(2)
        result = note.strip().split(',')
        result.pop()
        for i, item in enumerate(result):
            current_time += 240.0/(BPM*beat)
            if(item != ''):
                note_time.append(current_time)
                # print(item,current_time)
    # print(note_time)
    return note_time

if __name__ == "__main__":
    parse_note_time(r"MIRROR of MAGIC\maidata.txt",5)