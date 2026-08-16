import pretense


def test_import_helpers_do_not_leak_into_the_package_namespace() -> None:
    assert not hasattr(pretense, "PackageNotFoundError")
    assert not hasattr(pretense, "version")
