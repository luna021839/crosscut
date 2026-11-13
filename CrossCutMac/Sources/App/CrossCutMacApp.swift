import SwiftUI

@main
struct CrossCutMacApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 980, minHeight: 680)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1_120, height: 760)
    }
}
