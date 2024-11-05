## Naming Convention:
`{feature_name}__{param1}_{param2}_{param3}`

- `param1` should usually be the symbol.
- `feature_name` should be PascalCase
- Ex: `LogReturn__ETHUSDT_1M`

## When to Make a Feature
- When you found some good feature that you want to share with others.
- You shouldn't be pushing a feature unless you have found it to be useful. Therefore, there shouldn't be mass-created features, such as the square of every existing feature.

## How to Create a Feature
1. **Write the file:** Implement `__init__`, `get_parents`, and `get_feature` (follow the `LogReturn` example).
2. **Return Parents in `get_parents`:** All parents that you use in `get_feature` should be returned in `get_parents` so that the engine can fetch that parent for you.
3. Make sure to inherit the `AbstractFeature` class.
3. **Add the feature in `__init__.py`.**
4. **Test it using `feature_factory`.**
