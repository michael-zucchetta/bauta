import logging
import sys, os
import traceback
import subprocess

class SystemUtils():

    def hasExtension(self, filen_path, extensions):
        if os.path.isfile(filen_path):
            filename_and_extension = os.path.splitext(filen_path)
            return len(filename_and_extension) == 2 and filename_and_extension[1] in [f'.{extension}' for extension in extensions]
        else:
            return False

    def imagesInFolder(self, folder):
        if not os.path.isdir(folder):
            return []
        else:
            return list(filter(lambda filename: self.hasExtension(os.path.join(folder, filename), ['png', 'jpg']), os.listdir(folder)))

    def makeDirIfNotExists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def tryToRun(self, method, validate_result, max_attempts=10):
        current_attempt = 0
        successful_run = False
        result = None
        while not successful_run:
            try:
                result = method()
                successful_run = validate_result(result)
            except BaseException as e:
                sys.stderr.write(traceback.format_exc())
                current_attempt += 1
                sys.stderr.write(f"Error, trying again: {e}")
                if(current_attempt > max_attempts):
                    raise ValueError('It is not possible run method.', e)
        return result

    def getLogger(self, object_owner, level=logging.INFO):
        class_name = type(object_owner).__name__
        handlers = [logging.StreamHandler()]
        logging.basicConfig(level=level, handlers=handlers)
        return logging.getLogger(class_name)
