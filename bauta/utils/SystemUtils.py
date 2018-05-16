import logging
import sys, os
import re
import subprocess
import traceback
import shutil

class SystemUtils():

    def hasExtension(self, filen_path, extensions):
        if os.path.isfile(filen_path):
            filename_and_extension = os.path.splitext(filen_path)
            return len(filename_and_extension) == 2 and filename_and_extension[1] in [f'.{extension}' for extension in extensions]
        else:
            return False

    def imagesInFolder(self, folder, regex=None):
        if not os.path.isdir(folder):
            return []
        elif regex is not None:
            return [file_path for file_path in os.listdir(folder) if re.search(regex, file_path)]
        else:
            return list(filter(lambda filename: self.hasExtension(os.path.join(folder, filename), ['png', 'jpg', 'jpeg']), os.listdir(folder)))

    def makeDirIfNotExists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def removeFilesFromDir(self, dir_path):
        if os.path.exists(dir_path):
            [os.remove(os.path.join(dir_path, file_to_remove)) for file_to_remove in os.listdir(dir_path)]

    def tryToRun(self, method, validate_result, max_attempts=10):
        current_attempt = 0
        successful_run = False
        result = None
        call_stack = traceback.format_exc()
        while not successful_run:
            try:
                result = method()
                successful_run = validate_result(result)
                if not successful_run:
                    sys.stderr.write(traceback.format_exc())
                    raise ValueError(f'Validation did not succeed')
            except BaseException as e:
                sys.stderr.write(traceback.format_exc())
                current_attempt += 1
                sys.stderr.write(f"Error, trying again: {e}")
                if(current_attempt > max_attempts):
                    raise ValueError('It is not possible run method.', e)
        return result

    def rm(self, path):
        if os.path.isfile(path) :
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def getLogger(self, object_owner, level=logging.INFO):
        class_name = type(object_owner).__name__
        handlers = [logging.StreamHandler()]
        logging.basicConfig(level=level, handlers=handlers)
        return logging.getLogger(class_name)
