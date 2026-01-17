[app]
title = AI Game Generator
package.name = aigamegen
package.domain = org.groq
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = 1.0
requirements = python3,kivy
orientation = portrait
fullscreen = 1

[app:android]
android.api = 31
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.permissions = INTERNET
android.accept_sdk_license = True
