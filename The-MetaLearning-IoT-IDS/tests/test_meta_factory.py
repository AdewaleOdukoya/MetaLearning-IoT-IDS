from src.models.meta_factory import get_meta_model

model = get_meta_model(

    model_name="PROTONET",

    input_dim=46,

)

print(model)

print()

print("Meta Factory Test Passed.")