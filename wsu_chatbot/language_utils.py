import fasttext
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class LanguageDetector:
    _instance = None  # Class variable for implementing the singleton pattern

    def __new__(cls, model_path='C:/Users/manus/fastText/lid.176.bin'):
        """
        Ensures that only one instance of the LanguageDetector is created (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(LanguageDetector, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        """
        Load the FastText language detection model.
        """
        try:
            self.model = fasttext.load_model(self.model_path)
            logging.info("FastText model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load FastText model from {self.model_path}: {str(e)}")
            raise

    def detect_language_ft(self, text):
        """
        Detect the language of a given text using the FastText model.

        :param text: Input text for language detection
        :return: Detected language code or None if detection fails
        """
        text = self.sanitize_input(text)
        if not self.model:
            logging.error("FastText model is not loaded.")
            raise Exception("Model not loaded")
        try:
            predictions = self.model.predict(text, k=1)
            logging.info(f"Predictions: {predictions}")
            language_code = predictions[0][0].replace("__label__", "")
            return self.map_language_code(language_code)
        except Exception as e:
            logging.error(f"Failed to detect language for text '{text}': {str(e)}")
            return None

    def map_language_code(self, code):
        """
        Map FastText language codes to a friendlier language code format.

        :param code: FastText language code
        :return: Mapped language code or 'unknown_language' if not found
        """
        lang_map = {
            'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'et': 'et_EE',
            'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN', 'hi': 'hi_IN', 'it': 'it_IT', 'ja': 'ja_XX',
            'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT', 'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP',
            'nl': 'nl_XX', 'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN',
            'zh': 'zh_CN', 'af': 'af_ZA', 'az': 'az_AZ', 'bn': 'bn_IN', 'fa': 'fa_IR', 'he': 'he_IL',
            'hr': 'hr_HR', 'id': 'id_ID', 'ka': 'ka_GE', 'km': 'km_KH', 'mk': 'mk_MK', 'ml': 'ml_IN',
            'mn': 'mn_MN', 'mr': 'mr_IN', 'pl': 'pl_PL', 'ps': 'ps_AF', 'pt': 'pt_XX', 'sv': 'sv_SE',
            'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN', 'th': 'th_TH', 'tl': 'tl_XX', 'uk': 'uk_UA',
            'ur': 'ur_PK', 'xh': 'xh_ZA', 'gl': 'gl_ES', 'sl': 'sl_SI'
        }
        return lang_map.get(code, 'unknown_language')

    @staticmethod
    def sanitize_input(text):
        """
        Remove leading and trailing whitespace from the input text.

        :param text: Input text to sanitize
        :return: Sanitized text
        """
        return text.strip()

    def generate_response(self, text):
        """
        Generate a placeholder response based on detected language.

        :param text: Input text for language detection and response generation
        :return: Placeholder response or error message
        """
        language = self.detect_language_ft(text)
        if not language:
            return "Error: Unable to detect language."
        response = f"This is a placeholder response in language: {language}"
        logging.info(f"Generated response: {response}")
        return response

# Usage example
detector = LanguageDetector()
