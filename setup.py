############################################################################################
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
############################################################################################
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_tensorflow",
    version="1.3.0",
    author="Hasan Rafiq",
    description="""Build Low Code Automated Tensorflow, What-IF explainable models in just 3 lines of code. To make Deep Learning on Tensorflow absolutely easy for the masses with its low code framework and also increase trust on ML models through What-IF model explainability.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/rafiqhasan/auto-tensorflow",
    packages = [
        "auto_tensorflow"
    ],
    include_package_data=True,
    install_requires=[
        "keras-tuner==1.0.4",
        "tensorflow==2.6.2",
        "tensorflow_hub==0.12.0",
        "tensorflow_text==2.6.0",
        "tfx==1.4.0",
        "witwidget==1.8.0",
        "tensorflow-metadata==1.4.0",
        "ipython==7.29.0",
        "tensorflow-estimator==2.6.0",
        "joblib==0.14.1",
        "tensorboard-plugin-wit==1.8.0",
        "tensorboard-data-server==0.6.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
