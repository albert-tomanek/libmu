project('libmu', 'vala', 'c')

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
	],
	dependencies: [
		dependency('glib-2.0'),
		dependency('gobject-2.0'),
	],
	install: true,
)

main = executable('mu_test', 'test.vala', link_with: lib, dependencies: [dependency('glib-2.0'), dependency('gobject-2.0')], install: true)
