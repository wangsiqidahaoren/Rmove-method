import os
from common import Commonjavaextractor-0.0.1
from my_extractor import Extractor

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
# EXTRACTION_API = 'https://po3g2dx2qa.execute-api.us-east-1.amazonaws.com/production/extractmethods'
base_path = 'C:\\Users\\Dell\\Desktop\\jmeter'


class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([],'')
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,jar_path=JAR_PATH,max_path_length=self.config.MAX_PATH_LENGTH-1,max_path_width=2)
        # self.path_extractor = Extractor(config, EXTRACTION_API, self.config.MAX_PATH_LENGTH, max_path_width=2)

    @staticmethod
    def read_file(input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self):
        input_filename = 'Input.java'
        print('Serving')
        while True:
            print('Modify the file: "' + input_filename + '" and press any key when ready, or "q" / "exit" to exit')
            user_input = input()
            if user_input.lower() in self.exit_keywords:
                print('Exiting...')
                return
            user_input = ' '.join(self.read_file(input_filename))
            try:
                # predict_lines, pc_info_dict = self.path_extractor.extract_paths(user_input)
                predict_lines, pc_info_dict = self.path_extractor.extract_paths(input_filename)

            except ValueError:
                continue
            model_results = self.model.predict(predict_lines)

            prediction_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS)
            for index, method_prediction in prediction_results.items():
                print('Original name:\t' + method_prediction.original_name)
                if self.config.BEAM_WIDTH == 0:
                    print('Predicted:\t%s' % [step.prediction for step in method_prediction.predictions])
                    for timestep, single_timestep_prediction in enumerate(method_prediction.predictions):
                        print('Attention:')
                        print('TIMESTEP: %d\t: %s' % (timestep, single_timestep_prediction.prediction))
                        for attention_obj in single_timestep_prediction.attention_paths:
                            print('%f\tcontext: %s,%s,%s' % (
                                attention_obj['score'], attention_obj['token1'], attention_obj['path'],
                                attention_obj['token2']))
                else:
                    print('Predicted:')
                    for predicted_seq in method_prediction.predictions:
                        print('\t%s' % predicted_seq.prediction)

    def my_predict(self):
        li = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                filePath = os.path.splitext(file)
                if filePath[0] == '.java' or filePath[1] == '.java':
                    li.append(os.path.join(root, file))
        # print("--------------------------------------")
        # print(li, len(li))
        # print("--------------------------------------")
        for input_filename in li:
            print('Serving ' + input_filename)
            # user_input = ' '.join(self.read_file(input_filename))  # 将文件内每行内容用空格分隔
            try:
                predict_lines, pc_info_dict = self.path_extractor.extract_paths(input_filename)
            except ValueError:
                # print(ValueError)
                continue
                # predict_lines是个list 大小为input.java中的函数数量 元素类型为str，元素内容是由多个str(pc_info)组成的一个str
                # str(pc_info):return '%s,%s,%s' % (self.token1, self.shortPath, self.token2)
            model_results = self.model.predict(predict_lines, input_filename)

            prediction_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS)

            for index, method_prediction in prediction_results.items():
                print('Original name:\t' + method_prediction.original_name)
                if self.config.BEAM_WIDTH == 0:
                    print('Predicted:\t%s' % [step.prediction for step in method_prediction.predictions])
                    # Predicted:      ['reverse']
                    # for timestep, single_timestep_prediction in enumerate(method_prediction.predictions):
                        # print('Attention:')
                        # print('TIMESTEP: %d\t: %s' % (timestep, single_timestep_prediction.prediction))
                        # for attention_obj in single_timestep_prediction.attention_paths:
                        #     print('%f\tcontext: %s,%s,%s' % (
                        #         attention_obj['score'], attention_obj['token1'], attention_obj['path'],
                        #         attention_obj['token2']))
                else:
                    print('Predicted:')
                    for predicted_seq in method_prediction.predictions:
                        print('\t%s' % predicted_seq.prediction)
