---
title: "VS2013配置Caffe卷积神经网络工具（64位Windows 7）——准备依赖库"
category: "Computer Vision"
---

<p>2014年4月的时候自己在公司就将Caffe移植到Windows系统了，今年自己换了台电脑，想在家里也随便跑跑，本来也装了Ubuntu可以很方便的配置好，无奈在家的风格是&ldquo;娱乐的时候抽空学习&rdquo;，所以移植到Windows还是很有必要的。但是，公司禁止将公司内部资料带出，很多地方又都忘记了，周末磨了一天终于移植完，本篇为记录将Caffe移植至Windows7 x64系统下的一些关键步骤。第一步先看看这老些依赖库怎么搞。</p>
<p>在真正开始编译各依赖库之前，需要准备一些必备工具：</p>
<p>首先当然是VS2013，下载地址：<a href="https://www.visualstudio.com/">https://www.visualstudio.com/</a></p>
<p>其次是CMake工具，下载地址：<a href="http://www.cmake.org/download/">http://www.cmake.org/download/</a></p>
<h3>1. Boost</h3>
<p>下载地址：http://www.boost.org/</p>
<p>编译方法：</p>
<ol>
<li>运行Visual Studio Tools中的VS2013 x64 Cross Tools Command Prompt终端工具</li>
<li>从终端进入boost库所在目录，如D:\LIBS\boost<em>1</em>57_0</li>
<li>运行bootstrap.bat生产64位的bjam.exe</li>
<li>输入命令进行编译。静态库：<code>bjam --build-type=complete&nbsp;toolset=msvc-12.0 threading=multi link=static address-model=64</code> ，共享库：<code>bjam --build-type=complete&nbsp;toolset=msvc-12.0 threading=multi link=shared address-model=64</code></li>
</ol>
<h3>2. OpenCV</h3>
<p>下载地址：<a href="http://opencv.org/downloads.html">http://opencv.org/downloads.html</a> 本文中使用的是2.4.10版本</p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; 下载后的安装包中有已编译好的库，可直接引用，如D:\LIBS\opencv\build\x64\vc12</p>
<h3>3. OpenBlas</h3>
<p>下载地址：<a href="http://sourceforge.net/projects/openblas/files/">http://sourceforge.net/projects/openblas/files/</a></p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; OpenBlas库在windows上编译起来比较复杂，这里给出的下载地址是一个已编译好的压缩包OpenBLAS-v0.2.14-Win32.zip (12.1 MB)，直接提供了<code>./bin</code> <code>./include</code> <code>./lib</code>路径</p>
<h3>4. CUDA</h3>
<p>下载地址：<a href="https://developer.nvidia.com/cuda-downloads">https://developer.nvidia.com/cuda-downloads</a></p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; 与OpenCV类似，安装好后直接有已编译好的库。如C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include和C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\x64</p>
<p>以上依赖库比较常用，最好将该它们的include和lib加入到Windows的环境变量中，方便后续的库的编译</p>
<h3>5. GFlags</h3>
<p>下载地址：<a href="https://github.com/gflags/gflags">https://github.com/gflags/gflags</a></p>
<p>编译方法：</p>
<ol>
<li>启动CMake(cmake-gui)工具</li>
<li>设置source code路径和build路径</li>
<li>单击Configure按钮，并选择Visual Studio 12 2013 Win64编译器</li>
<li>更新完成后，勾选中BUILD<em>SHARED</em>LIBS和BUILD<em>STATIC</em>LIBS</li>
<li>单击Generate按钮，生成VS工程</li>
<li>打开刚刚生成的VS工程，build其中的ALL_BUILD工程，注意选择x64模式，并分别生成Debug和Release下的库</li>
<li>编译成功后，在工程路径下会生成bin、include、lib三个文件夹</li>
</ol>
<h3>6. GLog</h3>
<p>下载地址：<a href="https://github.com/google/glog">https://github.com/google/glog</a></p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; 该工程中包含VS工程google-glog.sln，打开直接编译即可，同样注意Solution Platform选择x64模式，并分别生成Debug和Release下的库</p>
<h3>7. LevelDB</h3>
<p>下载地址：<a href="https://github.com/bureau14/leveldb">https://github.com/bureau14/leveldb</a></p>
<p>&nbsp; &nbsp; &nbsp; 这里没有选择官方的https://github.com/google/leveldb是由于官方版本移除了CMake的依赖项，自己写起来比较复杂</p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; 与编译GFlags方法类似，唯一需要注意的地方是将CMakeLists.txt中第82行的<code>-DSNAPPY</code>注释掉，否则需要依赖Snappy库，其实并不绝对需要，为了简单起见将此库去掉。另外Leveldb依赖于boost库，如果没有将boost库添加至环境变量，可能需要手动进行设置。</p>
<h3>8. LMDB</h3>
<p>下载地址：<a href="https://github.com/clibs/lmdb">https://github.com/clibs/lmdb</a></p>
<p>编译方法：</p>
<ol>
<li>解压压缩包到某路径，例如D:\CODE\CXX\mdb-mdb</li>
<li>在VS2013中新建工程，FILE --&gt; New --&gt; Project From Existing Code..</li>
<li>选取源码所在路径，并给工程起名</li>
<li>单击next按钮后选择Project type为Console application project</li>
<li>将Solution Platform修改为x64模式</li>
<li>注意将工程的输出改为静态库，右键单击工程 --&gt; property --&gt; Configuration Properties --&gt; General --&gt; Project Default --&gt; Configureation Type --&gt; Static library (.lib)</li>
<li>其中一个.c文件中包含了unistd.h，为了解决这个问题需要准备三个文件unistd.h、getopt.h、getopt.c。unistd.h可以参考<a href="http://stackoverflow.com/questions/341817/is-there-a-replacement-for-unistd-h-for-windows-visual-c">http://stackoverflow.com/questions/341817/is-there-a-replacement-for-unistd-h-for-windows-visual-c</a>解决。另外两个可以从<a href="http://ieng6.ucsd.edu/~cs12x/vc08install/getopt9.zip">http://ieng6.ucsd.edu/~cs12x/vc08install/getopt9.zip</a>下载</li>
<li>最后编译即可</li>
</ol>
<h3>9. ProtoBuf</h3>
<p>下载地址：<a href="https://github.com/google/protobuf">https://github.com/google/protobuf</a></p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; 压缩包里有一个叫vsprojects的文件夹，其中有现成的VS工程，可以用来直接编译，也需要注意将Solution Platform修改为x64模式</p>
<h3>10. HDF5</h3>
<p>下载地址：<a href="http://www.hdfgroup.org/ftp/HDF5/current/src/CMake/hdf518-CMakeWindows.zip">http://www.hdfgroup.org/ftp/HDF5/current/src/CMake/hdf518-CMakeWindows.zip</a></p>
<p>编译方法：</p>
<p>&nbsp; &nbsp; &nbsp; 解压后，在VS2013 x64 Cross Tools Command Prompt终端工具中运行build-VS2013-64.bat即可。</p>
<h3>整理头文件和库文件</h3>
<p>将5-10的头文件和编译后的库统一整理到一个3rdparty文件夹下，其中包含两个文件夹include和lib。include文件夹下包含gflags、glog、google、hdf5、leveldb、lmdb六个文件夹。</p>
<p>gflags的头文件来自于生成的VS工程目录中的include文件夹；</p>
<p>glog的头文件来自于VS工程目录中的src\windows文件夹；</p>
<p>google中是protobuf的头文件，来自于压缩包中的src\google文件夹；</p>
<p>hdf5来自于压缩包中的CMake\hdf5-1.8.14\src文件夹，保留.h文件即可；</p>
<p>leveldb的头文件来自于压缩包的include文件夹；</p>
<p>lmdb的头文件来自于压缩包中的libraries\liblmdb文件夹，保留.h文件即可 lib文件夹中的.lib文件直接从编译好的工程目录下拷贝即可。</p>
<p>注意debug模式下的.lib文件的文件名修改为xxxd.lib形式。至此，caffe需要的各项依赖库已经准备完毕。</p>