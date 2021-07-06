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
    version="1.0.1",
    author="Hasan Rafiq",
    description="Build automated ML models using Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/rafiqhasan/auto-tensorflow",
    packages = [
        "auto_tensorflow"
    ],
    include_package_data=True,
    install_requires=[
        "keras-tuner==1.0.1",
        "tensorflow==2.5.0",
        "tensorflow_hub==0.12.0",
        "tfx==0.29.0",
        "witwidget==1.8.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
