import os
from os import path
import json
import xpinyin
from xpinyin import Pinyin

audio_files = ""

audio_files_fp = "./audio_files"
set1_transcript_fp = 'set1_transcript.json'


def index():
	wavindex = {}
	total = 0
	for i in os.listdir(audio_files_fp):
		i2fp = audio_files_fp + '/' + i
		for i2 in os.listdir(i2fp):
			i3fp = i2fp  + '/' + i2
			for i3 in os.listdir(i3fp):
				wavfp = i3fp + '/' + i3
				wavindex[i3] = wavfp
				total += 1     # 50384
	print(total)
	return wavindex
	
		
		
def parse_set1_transcript():
	wavindex = index()
	#print(wavindex)
	savef = open('audio_files_index.txt', 'w')
	flush = 100
	cnt = 0
	py = Pinyin()
	with open(set1_transcript_fp, 'r') as r:
		transcript_json  = json.load(r)
		for wavobj in transcript_json:
			text_py =  py.get_pinyin(wavobj['text'].replace(" ", ''), ' ', tone_marks='numbers')
			#print(text_py, wavobj['text'].encode('utf8'), wavobj['file'])
			savef.write(text_py.encode('utf8') + "###" + wavobj['text'].encode('utf8') + '###' + wavindex.get(wavobj['file'], "NOTFOUND") + '\n')
			cnt += 1
			if cnt % flush == 0:
				savef.flush()
		
			#print(wavobj['text'], wavobj['file'])
		#print(len(transcript_json))    # 50902
	savef.close()

def main():
	parse_set1_transcript()
	
if __name__ == "__main__":
	main()