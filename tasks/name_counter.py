from tasks.preprocess import PROCESSED_DATA_TYPE


class NameCounter:
    def __init__(self,
                 data: PROCESSED_DATA_TYPE):
        self.data = data
        self.name_counter = self._count_names()

    def __str__(self):
        print(self.to_json())

    def _count_names(self) -> list[list[str or int]]:
        text = ' '.join([' '.join(s) for s in self.data['Question 1']['Processed Sentences']])
        name_counter = []
        for person in self.data['Question 1']['Processed Names']:
            full_name = ' '.join(person[0])
            names_to_count = person[0] + [word for nickname in person[1] for word in nickname]
            counter = 0
            for name in set(names_to_count):
                counter += text.count(name)
            if counter > 0:
                name_counter.append([full_name, counter])
        name_counter.sort(key=lambda x: x[0])
        return name_counter

    def to_json(self):
        return {
            "Question 3": {
                "Name Mentions": self.name_counter,
            }
        }
