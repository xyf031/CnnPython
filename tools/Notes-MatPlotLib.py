

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))

plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
plt.plot(x,z,"b--",label="$cos(x^2)$")  # 可在当前画布中重复绘图

plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim(-1.2,1.2)
plt.legend()

plt.show()

通过figsize参数可以指定绘图对象的宽度和高度，单位为英寸；
dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80。
因此本例中所创建的图表窗口的宽度为8*80 = 640像素。

但是用工具栏中的保存按钮保存下来的png图像的大小是800*400像素。
这是因为保存图表用的函数savefig使用不同的DPI配置，savefig函数也有一个dpi参数，
如果不设置的话，将使用matplotlib配置文件中的配置，此配置可以通过如下语句进行查看
import matplotlib
matplotlib.rcParams["savefig.dpi"]

在IPython中输入 "plt.plot?" 可以查看格式化字符串的详细配置。


# ---------- set_* ----------
line, = plt.plot(x, x*x)  # plot返回一个列表，通过line,获取其第一个元素
line.set_antialiased(False)  # 调用Line2D对象的 set_* 方法设置属性值, 关闭对象的反锯齿效果

lines = plt.plot(x, np.sin(x), x, np.cos(x))  # 同时绘制sin和cos两条曲线，lines是一个有两个Line2D对象的列表
plt.setp(lines, color="r", linewidth=2.0)  # 调用 setp 函数同时配置多个Line2D对象的多个属性值


# ---------- get_* ----------
line.get_linewidth()  # 1.0
plt.getp(lines[0], "color") # 返回color属性 'r '
plt.getp(lines[1]) # 输出全部属性
# alpha = 1.0
# animated = False
# antialiased or aa = True
# axes = Axes(0.125,0.1;0.775x0.8)


# ---------- figure ----------
matplotlib的整个图表为一个Figure对象，可以通过plt.gcf函数获取当前的绘图对象：
f = plt.gcf()
plt.getp(f)
# alpha = 1.0
# animated = False


# ---------- subplot ----------
subplot将绘图区等分为 nRows行 * nCols列 个子区域，然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1。
如果numRows，numCols和plotNum这三个数都小于10的话，可以把它们缩写为一个整数，例如 subplot(323) 和 subplot(3,2,3) 是相同的。
subplot在plotNum指定的区域中创建一个轴对象。如果新创建的轴和之前创建的轴重叠的话，之前的轴将被删除。

subplot(numRows, numCols, plotNum)

如果希望某个轴占据整个行或者列的话，可以如下调用subplot：
plt.subplot(221)  # 第一行的左图
plt.subplot(222)  # 第一行的右图
plt.subplot(212)  # 第二整行
plt.show()

可以调用subplots_adjust函数，它有left, right, bottom, top, wspace, hspace等几个关键字参数，
这些参数的值都是0到1之间的小数，它们是以绘图区域的宽高为1进行正规化之后的坐标或者长度。


# ---------- rc_params ----------
配置文件的读入可以使用 rc_params 函数，它返回一个配置字典：

matplotlib.rc_params()
{'agg.path.chunksize': 0,
 'axes.axisbelow': False,
 'axes.edgecolor': 'k',
 'axes.facecolor': 'w',
 ... ...

在matplotlib模块载入的时候会调用rc_params，并把得到的配置字典保存到rcParams变量中：

matplotlib.rcParams
{'agg.path.chunksize': 0,
'axes.axisbelow': False, ...

matplotlib将使用rcParams中的配置进行绘图。用户可以直接修改此字典中的配置，所做的改变会反映到此后所绘制的图中。
恢复缺省设置：
matplotlib.rcdefaults()


# ---------- rc_params ----------
mypl = pl.subplot(111, axisbg='#111032')
pl.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
mypl.spines['right'].set_color('none')
mypl.spines['top'].set_color('none')
mypl.spines['left'].set_color('none')
mypl.spines['bottom'].set_color('none')
mypl.set_xticks([])
mypl.set_yticks([])

mypl.axis([-200, 200, Mercator(-86), Mercator(86)])
mypl.plot(xPlot[i], yPlot[i], '.', color=c, alpha=0.5, markersize=10*citySize[i]/5000000)
mypl.plot(xPlot[i], yPlot[i], 'w.', alpha=0.95, markersize=1)
mypl.fill_between(tmpX, tmpY, 0, facecolor=fillc, edgecolor='')
mypl.text(np.mean(xPlot), np.mean(yPlot), name, fontsize=2, color='0.6',
                  horizontalalignment='center', verticalalignment='center')

pl.savefig('WorldMap.png', dpi=300)
pl.show()


# ---------- rc_params ----------


# ---------- rc_params ----------


# ---------- rc_params ----------


# ---------- rc_params ----------


# ---------- rc_params ----------



#################################################### dir(matplotlib)
RcParams
Verbose
__builtins__
__doc__
__file__
__name__
__package__
__path__
__version__
__version__numpy__
_cm
_cntr
_create_tmp_config_dir
_delaunay
_deprecated_ignore_map
_deprecated_map
_get_cachedir
_get_config_or_cache_dir
_get_configdir
_get_data_path
_get_data_path_cached
_get_home
_get_xdg_cache_dir
_get_xdg_config_dir
_havedate
_image
_is_writable_dir
_mathtext_data
_path
_png
_pylab_helpers
_python24
_tri
_use_error_msg
absolute_import
afm
artist
ascii
axes
axis
backend_bases
backends
bad_pyparsing
bezier
blocking_input
byte2str
cbook
checkdep_dvipng
checkdep_ghostscript
checkdep_inkscape
checkdep_pdftops
checkdep_ps_distiller
checkdep_tex
checkdep_usetex
checkdep_xmllint
cm
collections
colorbar
colors
compare_versions
compat
container
contour
converter
dates
dateutil
default
defaultParams
default_test_modules
delaunay
distutils
docstring
dviread
f
figure
font_manager
fontconfig_pattern
ft2font
get_backend
get_cachedir
get_configdir
get_data_path
get_example_data
get_home
get_py2exe_datafiles
gridspec
image
interactive
is_interactive
is_string_like
key
legend
legend_handler
lines
major
markers
mathtext
matplotlib_fname
minor1
minor2
mlab
numpy
offsetbox
os
patches
path
print_function
projections
pyparsing
pyplot
quiver
rc
rcParams
rcParamsDefault
rcParamsOrig
rc_context
rc_file
rc_file_defaults
rc_params
rc_params_from_file
rcdefaults
rcsetup
re
s
scale
shutil
spines
stackplot
streamplot
subprocess
sys
table
tempfile
test
texmanager
text
textpath
ticker
tight_bbox
tk_window_focus
tmp
transforms
tri
units
use
validate_backend
validate_toolbar
verbose
warnings
widgets



#################################################### dir(matplotlib.pyplot)
pl(dir(matplotlib.pyplot))
Annotation
Arrow
Artist
AutoLocator
Axes
Button
Circle
Figure
FigureCanvasBase
FixedFormatter
FixedLocator
FormatStrFormatter
Formatter
FuncFormatter
GridSpec
IndexLocator
Line2D
LinearLocator
Locator
LogFormatter
LogFormatterExponent
LogFormatterMathtext
LogLocator
MaxNLocator
MultipleLocator
Normalize
NullFormatter
NullLocator
PolarAxes
Polygon
Rectangle
ScalarFormatter
Slider
Subplot
SubplotTool
Text
TickHelper
Widget
__builtins__
__doc__
__file__
__name__
__package__
_autogen_docstring
_backend_mod
_backend_selection
_imread
_imsave
_interactive_bk
_pylab_helpers
_setp
_setup_pyplot_info_docstrings
_show
_string_to_bool
acorr
annotate
arrow
autoscale
autumn
axes
axhline
axhspan
axis
axvline
axvspan
bar
barbs
barh
bone
box
boxplot
broken_barh
cla
clabel
clf
clim
close
cm
cohere
colorbar
colormaps
colors
connect
contour
contourf
cool
copper
csd
dedent
delaxes
disconnect
docstring
draw
draw_if_interactive
errorbar
eventplot
figaspect
figimage
figlegend
fignum_exists
figtext
figure
fill
fill_between
fill_betweenx
findobj
flag
gca
gcf
gci
get
get_backend
get_cmap
get_current_fig_manager
get_figlabels
get_fignums
get_plot_commands
get_scale_docs
get_scale_names
getp
ginput
gray
grid
hexbin
hist
hist2d
hlines
hold
hot
hsv
imread
imsave
imshow
interactive
ioff
ion
is_numlike
is_string_like
ishold
isinteractive
jet
legend
locator_params
loglog
margins
matplotlib
matshow
minorticks_off
minorticks_on
mlab
new_figure_manager
normalize
np
over
pause
pcolor
pcolormesh
pie
pink
plot
plot_date
plotfile
plotting
polar
print_function
prism
psd
pylab_setup
quiver
quiverkey
rc
rcParams
rcParamsDefault
rc_context
rcdefaults
register_cmap
rgrids
savefig
sca
scatter
sci
semilogx
semilogy
set_cmap
setp
show
silent_list
specgram
spectral
spring
spy
stackplot
stem
step
streamplot
subplot
subplot2grid
subplot_tool
subplots
subplots_adjust
summer
suptitle
switch_backend
sys
table
text
thetagrids
tick_params
ticklabel_format
tight_layout
title
tricontour
tricontourf
tripcolor
triplot
twinx
twiny
vlines
waitforbuttonpress
warnings
winter
xcorr
xkcd
xlabel
xlim
xscale
xticks
ylabel
ylim
yscale
yticks


#################################################### dir(plt.figure)
__class__
__delattr__
__dict__
__doc__
__format__
__getattribute__
__getstate__
__hash__
__init__
__module__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__setstate__
__sizeof__
__str__
__subclasshook__
__weakref__
_agg_filter
_alpha
_animated
_axobservers
_axstack
_cachedRenderer
_clipon
_clippath
_contains
_dpi
_gci
_get_axes
_get_dpi
_gid
_hold
_label
_lod
_make_key
_oid
_path_effects
_picker
_propobservers
_rasterized
_remove_method
_set_artist_props
_set_dpi
_set_gc_clip
_sketch
_snap
_suptitle
_tight
_tight_parameters
_transform
_transformSet
_url
_visible
add_axes
add_axobserver
add_callback
add_subplot
aname
artists
autofmt_xdate
axes
bbox
bbox_inches
callbacks
canvas
clear
clf
clipbox
colorbar
contains
convert_xunits
convert_yunits
delaxes
dpi
dpi_scale_trans
draw
draw_artist
eventson
figimage
figure
figurePatch
findobj
frameon
gca
get_agg_filter
get_alpha
get_animated
get_axes
get_children
get_clip_box
get_clip_on
get_clip_path
get_contains
get_default_bbox_extra_artists
get_dpi
get_edgecolor
get_facecolor
get_figheight
get_figure
get_figwidth
get_frameon
get_gid
get_label
get_path_effects
get_picker
get_rasterized
get_size_inches
get_sketch_params
get_snap
get_tight_layout
get_tightbbox
get_transform
get_transformed_clip_path_and_affine
get_url
get_visible
get_window_extent
get_zorder
ginput
have_units
hitlist
hold
images
is_figure_set
is_transform_set
legend
legends
lines
number
patch
patches
pchanged
pick
pickable
properties
remove
remove_callback
savefig
sca
set
set_agg_filter
set_alpha
set_animated
set_axes
set_canvas
set_clip_box
set_clip_on
set_clip_path
set_contains
set_dpi
set_edgecolor
set_facecolor
set_figheight
set_figure
set_figwidth
set_frameon
set_gid
set_label
set_lod
set_path_effects
set_picker
set_rasterized
set_size_inches
set_sketch_params
set_snap
set_tight_layout
set_transform
set_url
set_visible
set_zorder
show
subplotpars
subplots_adjust
suppressComposite
suptitle
text
texts
tight_layout
transFigure
update
update_from
waitforbuttonpress
zorder



#################################################### dir(matplotlib)



#################################################### dir(matplotlib)



#################################################### dir(matplotlib)

