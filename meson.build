project('libmu', 'vala', 'c')

if get_option('buildtype') == 'custom'
  add_project_arguments('-g',  language : 'c')
endif

lib = library('mu',
	version: '0.0.1',
	sources: [
		'src/array.vala',
		'src/shape.vala',
		'src/index.vala',
		'src/ops.vala',
		'src/ops_math.vala',

		'src/tinyndarray/broadcast.vala',
		'src/tinyndarray/reduce.vala',
		'src/tinyndarray/parallel.vala',
		'src/tinyndarray/dot.vala',
		'src/tinyndarray/concat.vala',
	],
	dependencies: [
		dependency('glib-2.0'),
		dependency('gobject-2.0'),
	],
	install: true,
)

main = executable('mu_test', 'test.vala', link_with: lib, dependencies: [dependency('glib-2.0'), dependency('gobject-2.0')], install: true)
