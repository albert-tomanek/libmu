project('libmu', 'vala', 'c')

lib = library('mu',
	sources: [
		'src/array.vala',
		'src/shape.vala',
		'src/ops.vala',
	],
	dependencies: [
		dependency('glib-2.0'),
		dependency('gobject-2.0'),
	],
	install: true,
)

main = executable('mu_test', 'test.vala', link_with: lib, dependencies: [dependency('glib-2.0'), dependency('gobject-2.0')], install: true)
