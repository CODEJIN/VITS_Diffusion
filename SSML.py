import math, hgtk
from enum import Enum
from html.parser import HTMLParser
from Pattern_Generator import Decompose

class Scale_Type(Enum):
    Replace= 0
    Add= 1
    Multiply= 2
    Min= 3
    Max= 4

step_ms = 256 / 22050 * 1000

class SSMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = {key.lower(): value for key, value in attrs}
        
        if tag == 'speak':
            self.patterns = []
        elif tag == 'voice':
            self.in_voice = True
            self.text = []
            self.duration_scale = [(Scale_Type.Multiply, 1.0)]  # <S> token
            self.log_f0_scale = [(Scale_Type.Add, 0.0)] # <S> token
            self.energy_scale = [(Scale_Type.Add, 0.0)] # <S> token
            self.current_speakers = [x.strip() for x in attrs['name'].split(',')] if 'name' in attrs.keys() else ['SelectStar_Female_01']
            self.current_weights = [float(x.strip()) for x in attrs['weight'].split(',')] if 'weight' in attrs.keys() else [1.0 / len(self.current_speakers)] * len(self.current_speakers)
            self.current_emotion = attrs['emotion'] if 'emotion' in attrs.keys() else 'Neutral'
            self.current_duration_scales = [(Scale_Type.Multiply, 1.0)] # Basic
            self.current_f0_scales = [(Scale_Type.Add, 0.0)]    # Basic
            self.current_energy_scales = [(Scale_Type.Add, 0.0)]    # Basic
        elif tag == 'prosody':
            if 'rate' in attrs.keys():
                scale_type, scale = self.Prosody_Parse(attrs['rate'])
                if scale_type != Scale_Type.Multiply:
                    scale = math.ceil(scale / step_ms)
                self.current_duration_scales.append((scale_type, scale))
            else:
                self.current_duration_scales.append(self.current_duration_scales[-1])
            self.current_f0_scales.append(self.Prosody_Parse(attrs['pitch']) if 'pitch' in attrs.keys() else self.current_f0_scales[-1])
            self.current_energy_scales.append(self.Prosody_Parse(attrs['volumn']) if 'volumn' in attrs.keys() else self.current_energy_scales[-1])
        elif tag == 'break':
            if not 'time' in attrs.keys():
                attrs['time'] = 500
            step = math.ceil(float(attrs['time']) / step_ms)
            self.text.append(' ')
            self.duration_scale.append((Scale_Type.Replace, step))
            self.log_f0_scale.append((Scale_Type.Replace, -5.0))
            self.energy_scale.append((Scale_Type.Multiply, 1.0))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ['speak', 'break']:
            return
        elif tag == 'voice':
            self.duration_scale.append((Scale_Type.Multiply, 1.0))  # <E> token
            self.log_f0_scale.append((Scale_Type.Add, 0.0)) # <E> token
            self.energy_scale.append((Scale_Type.Add, 0.0)) # <E> token
            self.patterns.append({
                'Decomposed': self.text,
                'Duration_Scale': self.duration_scale,
                'Log_F0_Scale': self.log_f0_scale,
                'Energy_Scale': self.energy_scale,
                'Speakers': self.current_speakers,
                'Weights': self.current_weights,
                'Emotion': self.current_emotion,
                })
            self.in_voice = False
        elif tag == 'prosody':
            self.current_duration_scales.pop()
            self.current_f0_scales.pop()
            self.current_energy_scales.pop()

    def handle_data(self, data):
        if not self.in_voice: return

        decomposed = Decompose(data)
        self.text.extend(decomposed)

        for letter in decomposed:
            scale = self.current_duration_scales[-1]
            if scale[0] == Scale_Type.Replace and (hgtk.checker.is_hangul(letter) or letter[-1] == '_'):
                scale = (scale[0], scale[1] // 3)   # Hangul use 3 letters
            self.duration_scale.append(scale)
        self.log_f0_scale.extend([self.current_f0_scales[-1]] * len(decomposed))
        self.energy_scale.extend([self.current_energy_scales[-1]] * len(decomposed))

    def Prosody_Parse(self, attr):
        if attr[0] == '=':
            return Scale_Type.Replace, float(attr[1:])
        elif attr[0] == '+':
            return Scale_Type.Add, float(attr[1:])
        elif attr[0] == '-':
            return Scale_Type.Add, -1.0 * float(attr[1:])
        elif attr[0] == '*':
            return Scale_Type.Multiply, float(attr[1:])
        elif attr[0] == '/':
            return Scale_Type.Multiply, 1.0 / float(attr[1:])
        elif attr[0] == '<':
            return Scale_Type.Max, float(attr[1:])
        elif attr[0] == '>':
            return Scale_Type.Min, float(attr[1:])
        else:
            return Scale_Type.Add, 0.0  # Ignored

    def feed(self, data):
        self.in_voice = False
        super().feed(data)

        return self.patterns

class SSMLChecker(HTMLParser):
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.tags.append(tag)

    def handle_endtag(self, tag):
        if self.tags[-1] == tag:
            self.tags.pop()
        else:
            self.is_valid = False

    def feed(self, data):
        self.tags = []
        self.is_valid = True
        super().feed(data)
        if len(self.tags) > 0:
            self.is_valid = False
        del self.tags
        return self.is_valid